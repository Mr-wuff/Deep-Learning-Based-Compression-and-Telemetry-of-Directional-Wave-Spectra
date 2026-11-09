"""
WaveSystem Set Parser V5 for ICWDS
==================================
A lightweight physics-seeded set parser for wave directional spectra.

Key changes from v4.8:
- dual proposal source: external physics proposals + learned seed proposals;
- multi-round relation message passing;
- slot set prediction with per-slot existence probabilities;
- no count-soft -> round -> top-N hard deletion;
- separate core and support mask heads;
- auxiliary wind-sea and wave-age heads for real weak supervision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WaveSystemSetParserV5Config:
    n_freqs: int = 47
    n_dirs: int = 72
    n_slots: int = 6
    p_phys_max: int = 24
    learned_seed_k: int = 8
    prop_feat_dim: int = 22
    width: int = 32
    depth: int = 4
    node_dim: int = 64
    slot_dim: int = 64
    msg_rounds: int = 3
    pair_feat_dim: int = 8
    seed_sigma_f: float = 0.085
    seed_sigma_t: float = 0.16  # fraction of pi
    prior_gain_core: float = 3.2
    prior_gain_support: float = 2.0
    min_exist_bias: float = -0.8

    @property
    def p_total(self) -> int:
        return self.p_phys_max + self.learned_seed_k

    def to_dict(self):
        return asdict(self)


class DepthwiseSeparable(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.dw = nn.Conv2d(ci, ci, 3, padding=1, groups=ci, bias=False)
        self.pw = nn.Conv2d(ci, co, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, co), co)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class PhysicsAwareModule(nn.Module):
    def __init__(self, ch: int, n_dirs: int):
        super().__init__()
        dirs = torch.linspace(0, 2 * math.pi, n_dirs + 1)[:n_dirs]
        self.n_dirs = n_dirs
        self.register_buffer("cos_d", torch.cos(dirs).view(1, 1, 1, n_dirs))
        self.register_buffer("sin_d", torch.sin(dirs).view(1, 1, 1, n_dirs))
        self.fuse = nn.Conv2d(ch + 3, ch, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.act = nn.GELU()

    def _phys_features(self, E):
        eps = 1e-8
        En = torch.nan_to_num(E, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0)
        En = En / (En.amax(dim=(2, 3), keepdim=True) + eps)
        nf = En.shape[2]
        rev_cumsum = torch.flip(torch.cumsum(torch.flip(En, dims=[2]), dim=2), dims=[2])
        col_sum = En.sum(dim=2, keepdim=True) + eps
        hf_tail = rev_cumsum / col_sum
        cx = (En * self.cos_d).sum(dim=3, keepdim=True)
        cy = (En * self.sin_d).sum(dim=3, keepdim=True)
        row_sum = En.sum(dim=3, keepdim=True) + eps
        dir_conc = torch.sqrt(cx ** 2 + cy ** 2) / row_sum
        dir_conc = dir_conc.expand(-1, -1, -1, self.n_dirs)
        fcoord = torch.linspace(0, 1, nf, device=E.device, dtype=E.dtype).view(1, 1, nf, 1)
        fc = (En * fcoord).sum(dim=2, keepdim=True) / col_sum
        spread = torch.sqrt((En * (fcoord - fc) ** 2).sum(dim=2, keepdim=True) / col_sum)
        spread = spread.expand(-1, -1, nf, -1)
        return torch.cat([hf_tail, dir_conc, spread], dim=1)

    def forward(self, h, E):
        return self.act(self.norm(self.fuse(torch.cat([h, self._phys_features(E)], dim=1))))


class RelationBlock(nn.Module):
    def __init__(self, dim: int, pair_dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(pair_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, node, pair_feat, valid):
        # node [B,P,D], pair_feat [B,P,P,F], valid [B,P]
        D = node.shape[-1]
        q = self.q(node)
        k = self.k(node)
        v = self.v(node)
        logits = torch.einsum("bpd,bqd->bpq", q, k) / math.sqrt(D)
        logits = logits + self.edge_bias(pair_feat).squeeze(-1)
        pair_valid = valid[:, :, None].bool() & valid[:, None, :].bool()
        logits = logits.masked_fill(~pair_valid, -30.0)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        msg = torch.einsum("bpq,bqd->bpd", attn, v)
        upd = self.update(torch.cat([node, msg], dim=-1))
        out = self.norm(node + upd)
        return out * valid[:, :, None]


class WaveSystemSetParserV5(nn.Module):
    def __init__(self, cfg: Optional[WaveSystemSetParserV5Config] = None):
        super().__init__()
        self.cfg = cfg or WaveSystemSetParserV5Config()
        c = self.cfg

        # Names intentionally match the v4.8 backbone for partial warm-start.
        self.stem = nn.Conv2d(4, c.width, 3, padding=1)
        self.stem_norm = nn.GroupNorm(min(8, c.width), c.width)
        self.physics = PhysicsAwareModule(c.width, c.n_dirs)
        self.blocks = nn.ModuleList([DepthwiseSeparable(c.width, c.width) for _ in range(c.depth)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.seed_head = nn.Sequential(
            nn.Conv2d(c.width, c.width, 3, padding=1), nn.GELU(), nn.Conv2d(c.width, 1, 1)
        )
        self.global_support_head = nn.Sequential(
            nn.Conv2d(c.width, c.width, 3, padding=1), nn.GELU(), nn.Conv2d(c.width, 1, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(c.width + c.prop_feat_dim, c.node_dim), nn.GELU(),
            nn.Linear(c.node_dim, c.node_dim), nn.GELU(),
        )
        self.node_keep = nn.Linear(c.node_dim, 1)
        self.relation = nn.ModuleList([
            RelationBlock(c.node_dim, c.pair_feat_dim) for _ in range(c.msg_rounds)
        ])
        self.edge_head = nn.Sequential(
            nn.Linear(4 * c.node_dim + c.pair_feat_dim, c.node_dim), nn.GELU(),
            nn.Linear(c.node_dim, c.node_dim), nn.GELU(), nn.Linear(c.node_dim, 1)
        )

        self.slot_queries = nn.Parameter(torch.randn(c.n_slots, c.slot_dim) * 0.02)
        self.slot_q = nn.Linear(c.slot_dim, c.node_dim, bias=False)
        self.slot_k = nn.Linear(c.node_dim, c.node_dim, bias=False)
        self.slot_v = nn.Linear(c.node_dim, c.slot_dim, bias=False)
        self.slot_update = nn.Sequential(
            nn.Linear(c.slot_dim * 2, c.slot_dim * 2), nn.GELU(), nn.Linear(c.slot_dim * 2, c.slot_dim)
        )
        self.slot_norm = nn.LayerNorm(c.slot_dim)
        self.exist_head = nn.Linear(c.slot_dim, 1)
        nn.init.constant_(self.exist_head.bias, c.min_exist_bias)
        self.count_head = nn.Sequential(
            nn.Linear(c.width + c.slot_dim, c.slot_dim), nn.GELU(), nn.Linear(c.slot_dim, c.n_slots + 1)
        )

        self.pixel_embed = nn.Conv2d(c.width, c.slot_dim, 1, bias=False)
        self.core_slot_proj = nn.Linear(c.slot_dim, c.slot_dim, bias=False)
        self.support_slot_proj = nn.Linear(c.slot_dim, c.slot_dim, bias=False)
        self.bg_head = nn.Sequential(
            nn.Conv2d(c.width, c.width, 3, padding=1), nn.GELU(), nn.Conv2d(c.width, 1, 1)
        )

        self.windsea_head = nn.Sequential(
            nn.Conv2d(c.width, c.width // 2, 3, padding=1), nn.GELU(), nn.Conv2d(c.width // 2, 1, 1)
        )
        self.wave_age_head = nn.Sequential(
            nn.Conv2d(c.width, c.width // 2, 3, padding=1), nn.GELU(), nn.Conv2d(c.width // 2, 1, 1)
        )

        self.register_buffer("fgrid", torch.linspace(0, 1, c.n_freqs).view(1, 1, c.n_freqs, 1))
        ang = torch.linspace(0, 2 * math.pi, c.n_dirs + 1)[:c.n_dirs]
        self.register_buffer("tgrid", ang.view(1, 1, 1, c.n_dirs))
        self.register_buffer("sin_grid", torch.sin(ang).view(1, 1, 1, c.n_dirs))
        self.register_buffer("cos_grid", torch.cos(ang).view(1, 1, 1, c.n_dirs))
        self._coord_cache = None

    def _coord_channels(self, B, device, dtype):
        if self._coord_cache is None:
            nf, nd = self.cfg.n_freqs, self.cfg.n_dirs
            fcoord = torch.linspace(0, 1, nf).view(1, 1, nf, 1).expand(1, 1, nf, nd)
            ang = torch.linspace(0, 2 * math.pi, nd + 1)[:nd].view(1, 1, 1, nd).expand(1, 1, nf, nd)
            self._coord_cache = torch.cat([fcoord, torch.sin(ang), torch.cos(ang)], dim=1)
        return self._coord_cache.to(device=device, dtype=dtype).expand(B, -1, -1, -1)

    @staticmethod
    def _safe_softmax(x, dim=-1):
        x = torch.nan_to_num(x, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)
        x = x - x.max(dim=dim, keepdim=True).values.detach()
        return torch.nan_to_num(torch.softmax(x, dim=dim), nan=0.0)

    def _periodic_max_pool(self, x, kernel=3):
        pad = kernel // 2
        xp = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
        yp = F.max_pool2d(xp, kernel_size=kernel, stride=1, padding=(pad, 0))
        return yp

    def _learned_proposals(self, E01, seed_logits, support_logits) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cfg
        B, _, H, W = seed_logits.shape
        seed = torch.sigmoid(seed_logits)
        support = torch.sigmoid(support_logits)
        pooled = self._periodic_max_pool(seed, 3)
        peaks = seed * (seed >= pooled - 1e-6).float()
        vals, idx = torch.topk(peaks.flatten(1), k=c.learned_seed_k, dim=1)
        rr = (idx // W).float() / max(H - 1, 1)
        cc = idx % W
        theta = cc.float() / W * (2 * math.pi)

        fg = self.fgrid.to(E01.dtype)
        tg = self.tgrid.to(E01.dtype)
        df = (fg - rr[:, :, None, None]) / c.seed_sigma_f
        dt = torch.atan2(
            torch.sin(tg - theta[:, :, None, None]),
            torch.cos(tg - theta[:, :, None, None]),
        ) / (math.pi * c.seed_sigma_t)
        gauss = torch.exp(-0.5 * (df * df + dt * dt))
        masks = (gauss * (0.15 + 0.85 * support)).clamp(0, 1)
        valid = (vals > 0.03).float()
        masks = masks * valid[:, :, None, None]

        total = E01.sum(dim=(2, 3)).clamp_min(1e-6)
        mass = torch.einsum("bchw,bphw->bp", E01, masks) / total
        peak = vals.clamp(0, 1)
        area = masks.mean(dim=(2, 3))
        sf = torch.full_like(mass, c.seed_sigma_f)
        st = torch.full_like(mass, c.seed_sigma_t)
        zeros = torch.zeros_like(mass)
        ones = torch.ones_like(mass)
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        quality = peak
        feats = torch.stack([
            mass, peak, area, rr, sin_t, cos_t, sf, st,
            area, rr, torch.full_like(mass, c.learned_seed_k / float(c.p_total)), quality,
            peak, zeros, ones * 0.5, mass, ones, ones, zeros, peak, ones, zeros,
        ], dim=-1)
        return masks, feats, valid

    def _node_pool(self, h, masks, valid):
        denom = masks.flatten(2).sum(dim=2).clamp_min(1.0)
        pooled = torch.einsum("bchw,bphw->bpc", h, masks) / denom[:, :, None]
        return pooled * valid[:, :, None]

    def _pair_features(self, feats):
        fi = feats[:, :, 3][:, :, None]; fj = feats[:, :, 3][:, None, :]
        d_f = (fi - fj).abs()
        si = feats[:, :, 4][:, :, None]; ci = feats[:, :, 5][:, :, None]
        sj = feats[:, :, 4][:, None, :]; cj = feats[:, :, 5][:, None, :]
        dot = (si * sj + ci * cj).clamp(-1 + 1e-5, 1 - 1e-5)
        d_t = torch.acos(dot) / math.pi
        mi = feats[:, :, 0][:, :, None]; mj = feats[:, :, 0][:, None, :]
        d_m = (mi - mj).abs()
        sfi = feats[:, :, 6][:, :, None]; sfj = feats[:, :, 6][:, None, :]
        sti = feats[:, :, 7][:, :, None]; stj = feats[:, :, 7][:, None, :]
        d_sf = (sfi - sfj).abs(); d_st = (sti - stj).abs()
        prom = torch.minimum(feats[:, :, 12][:, :, None], feats[:, :, 12][:, None, :])
        stripe = torch.maximum(feats[:, :, 13][:, :, None], feats[:, :, 13][:, None, :])
        qual = torch.minimum(feats[:, :, 11][:, :, None], feats[:, :, 11][:, None, :])
        return torch.stack([d_f, d_t, d_m, d_sf, d_st, prom, stripe, qual], dim=-1)

    def _slot_reasoning(self, node, node_keep, valid):
        B = node.shape[0]
        slots = self.slot_queries[None].expand(B, -1, -1)
        key = self.slot_k(node)
        value = self.slot_v(node)
        for _ in range(2):
            q = self.slot_q(slots)
            score = torch.einsum("bkd,bpd->bkp", q, key) / math.sqrt(key.shape[-1])
            score = score + torch.log(node_keep[:, None, :].clamp_min(1e-4))
            score = score.masked_fill(valid[:, None, :] <= 0, -30.0)
            attn = self._safe_softmax(score, dim=-1)
            ctx = torch.einsum("bkp,bpd->bkd", attn, value)
            slots = self.slot_norm(slots + self.slot_update(torch.cat([slots, ctx], dim=-1)))
        return slots

    def forward(self, x, prop_masks, prop_feats, prop_valid) -> Dict[str, torch.Tensor]:
        c = self.cfg
        B, _, H, W = x.shape
        prop_masks = torch.nan_to_num(prop_masks.float(), nan=0.0).clamp(0, 1)
        prop_feats = torch.nan_to_num(prop_feats.float(), nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5, 5)
        prop_valid = prop_valid.float().clamp(0, 1)

        h_in = torch.cat([x, self._coord_channels(B, x.device, x.dtype)], dim=1)
        h = F.gelu(self.stem_norm(self.stem(h_in)))
        E01 = ((x + 1.0) * 0.5).clamp(0, 1)
        h = h + self.physics(h, E01)
        for blk in self.blocks:
            h = h + blk(h)

        seed_logits = self.seed_head(h)
        global_support_logits = self.global_support_head(h)
        learned_masks, learned_feats, learned_valid = self._learned_proposals(E01, seed_logits, global_support_logits)

        masks = torch.cat([prop_masks, learned_masks], dim=1)
        feats = torch.cat([prop_feats, learned_feats], dim=1)
        valid = torch.cat([prop_valid, learned_valid], dim=1)

        node_pool = self._node_pool(h, masks, valid)
        node = self.node_mlp(torch.cat([node_pool, feats], dim=-1)) * valid[:, :, None]
        pair_feat = self._pair_features(feats)
        for block in self.relation:
            node = block(node, pair_feat, valid)

        node_keep_logit = self.node_keep(node).squeeze(-1).masked_fill(valid <= 0, -20.0)
        node_keep = torch.sigmoid(node_keep_logit) * valid

        # Pair merge probability is diagnostic/training supervision; set prediction itself is not hard-merged.
        P = node.shape[1]
        ni = node[:, :, None, :].expand(B, P, P, -1)
        nj = node[:, None, :, :].expand(B, P, P, -1)
        edge_in = torch.cat([ni, nj, (ni - nj).abs(), ni * nj, pair_feat], dim=-1)
        edge_logits = self.edge_head(edge_in).squeeze(-1)
        pair_valid = valid[:, :, None].bool() & valid[:, None, :].bool()
        eye = torch.eye(P, device=x.device, dtype=torch.bool)[None]
        edge_valid = pair_valid & (~eye)
        edge_logits = edge_logits.masked_fill(~edge_valid, 0.0)

        slots = self._slot_reasoning(node, node_keep, valid)
        exist_logit = self.exist_head(slots).squeeze(-1)
        exist_prob = torch.sigmoid(exist_logit)

        slot_key = self.slot_q(slots)
        node_slot_logits = torch.einsum("bpd,bkd->bpk", node, slot_key) / math.sqrt(node.shape[-1])
        node_slot_logits = node_slot_logits.masked_fill(valid[:, :, None] <= 0, -20.0)
        node_slot = self._safe_softmax(node_slot_logits, dim=-1) * valid[:, :, None]
        assign = node_slot * node_keep[:, :, None]
        prior_signal = torch.einsum("bpk,bphw->bkhw", assign, masks)
        prior_signal = prior_signal / prior_signal.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        prior_signal = prior_signal.clamp(0, 1)
        prior_expand = F.max_pool2d(prior_signal, 5, stride=1, padding=2)

        pix = self.pixel_embed(h)
        core_vec = self.core_slot_proj(slots)
        support_vec = self.support_slot_proj(slots)
        core_logits = torch.einsum("bkd,bdhw->bkhw", core_vec, pix) / math.sqrt(c.slot_dim)
        support_logits = torch.einsum("bkd,bdhw->bkhw", support_vec, pix) / math.sqrt(c.slot_dim)
        core_logits = core_logits + c.prior_gain_core * (prior_signal - 0.35)
        support_logits = support_logits + c.prior_gain_support * (prior_expand - 0.30) + 1.25 * (torch.sigmoid(global_support_logits) - 0.5)
        core_prob = torch.sigmoid(core_logits)
        support_prob = torch.sigmoid(support_logits)
        mask_prob = (support_prob * (0.35 + 0.65 * core_prob)).clamp(0, 1)

        slot_score = mask_prob * exist_prob[:, :, None, None]
        bg_logit = self.bg_head(h)
        bg_score = torch.sigmoid(bg_logit)
        scores = torch.cat([slot_score, bg_score], dim=1).clamp_min(1e-6)
        prob = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

        global_feat = self.global_pool(h).flatten(1)
        count_logits = self.count_head(torch.cat([global_feat, slots.mean(dim=1)], dim=-1))

        return {
            "prob": prob,
            "mask_prob": mask_prob,
            "core_prob": core_prob,
            "support_prob": support_prob,
            "exist_logit": exist_logit,
            "exist_prob": exist_prob,
            "count_logits": count_logits,
            "seed_logits": seed_logits,
            "global_support_logits": global_support_logits,
            "windsea_logits": self.windsea_head(h),
            "wave_age_pred": torch.sigmoid(self.wave_age_head(h)),
            "node_keep_logit": node_keep_logit,
            "node_slot_logits": node_slot_logits,
            "edge_logits": edge_logits,
            "edge_valid": edge_valid,
            "all_prop_masks": masks,
            "all_prop_valid": valid,
            "prior_signal": prior_signal,
        }


def transfer_v48_backbone(model: WaveSystemSetParserV5, state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Load shape-compatible v4.8 backbone tensors into V5."""
    target = model.state_dict()
    prefixes = ("stem.", "stem_norm.", "physics.", "blocks.")
    matched = {}
    for k, v in state_dict.items():
        kk = k
        for pfx in ("module.", "model.", "cnn.", "seg_model.", "seg."):
            if kk.startswith(pfx):
                kk = kk[len(pfx):]
        if kk.startswith(prefixes) and kk in target and tuple(v.shape) == tuple(target[kk].shape):
            matched[kk] = v
    model.load_state_dict(matched, strict=False)
    total_backbone = sum(v.numel() for k, v in target.items() if k.startswith(prefixes))
    loaded = sum(target[k].numel() for k in matched)
    return {
        "matched_tensors": len(matched),
        "backbone_numel_coverage": loaded / max(total_backbone, 1),
    }

# -----------------------------------------------------------------------------
# Archive/runtime helpers
# -----------------------------------------------------------------------------
def build_model(config: Optional[WaveSystemSetParserV5Config] = None) -> WaveSystemSetParserV5:
    """Construct the exact V5 segmentation architecture used by CNN.pt."""
    return WaveSystemSetParserV5(config or WaveSystemSetParserV5Config())


def load_cnn_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> WaveSystemSetParserV5:
    """
    Load the archived CNN.pt checkpoint.

    CNN.pt is expected to be the renamed V5 phase-C complete checkpoint:
      WaveSystemSetParserV5/checkpoints/phase_C/complete.pt
    """
    ck = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    cfg_dict = ck.get("config", {}) if isinstance(ck, dict) else {}
    allowed = set(WaveSystemSetParserV5Config.__dataclass_fields__)
    cfg = WaveSystemSetParserV5Config(**{k: v for k, v in cfg_dict.items() if k in allowed})
    model = WaveSystemSetParserV5(cfg)

    if isinstance(ck, dict):
        state = ck.get("model", ck.get("state_dict", ck))
    else:
        state = ck
    model.load_state_dict(state, strict=strict)
    return model


__all__ = [
    "WaveSystemSetParserV5Config",
    "WaveSystemSetParserV5",
    "transfer_v48_backbone",
    "build_model",
    "load_cnn_checkpoint",
]