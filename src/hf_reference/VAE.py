"""
V5-Aware Analytic-Centered Structured Semantic VAE (AC-SSVAE)
================================================================
Stage-2 feature extractor for ICWDS.

Design goals
------------
1. Consume WaveSystemSetParserV5 soft wave-system outputs instead of hard labels.
2. Use an analytic physical descriptor as the prior center, and learn only bounded
   semantic corrections.
3. Canonicalize each wave system before shape encoding so the free latent does not
   waste capacity on location, direction, scale, or energy.
4. Keep a small causal shape latent. The residual decoder is constructed as
       Delta(z, s) = F(z, s) - F(0, s)
   so z=0 exactly returns the semantic base.
5. Expose deterministic decoding APIs for quantization, intervention, and later
   30-byte packet design.

Internal semantic vector (9D)
-----------------------------
[log_energy,
 peak_frequency_01,
 sin_peak_direction,
 cos_peak_direction,
 frequency_spread_01,
 direction_spread_over_pi,
 f_theta_correlation,
 frequency_skew,
 direction_skew]

The packet-facing direction is still one circular quantity. sin/cos are only the
neural internal representation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class V5AwareACSSVAEConfig:
    n_freqs: int = 47
    n_dirs: int = 72
    n_slots: int = 6
    semantic_dim: int = 9
    shape_dim: int = 4
    context_dim: int = 24

    canonical_h: int = 32
    canonical_w: int = 48
    canonical_f_extent: float = 3.0
    canonical_t_extent: float = 3.0

    encoder_width: int = 48
    hidden_dim: int = 160
    decoder_width: int = 64

    # Bounded semantic correction magnitudes around analytic center.
    # [logE, f, sin/cos correction, spreads, rho, skews]
    delta_loge: float = 0.35
    delta_f: float = 0.08
    delta_angle_rad: float = math.radians(18.0)
    delta_log_spread: float = 0.45
    delta_rho_logit: float = 0.75
    delta_skew: float = 0.90

    min_spread_f: float = 0.010
    max_spread_f: float = 0.45
    min_spread_t: float = 0.015
    max_spread_t: float = 0.95
    max_abs_rho: float = 0.92
    max_abs_skew: float = 3.0

    shape_log_residual_scale: float = 2.25
    energy_log_den: float = math.log1p(47 * 72)
    eps: float = 1e-6

    def to_dict(self):
        return asdict(self)


class ConvNormAct(nn.Module):
    def __init__(self, ci: int, co: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, co), co)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch), ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch), ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _atanh_safe(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, stochastic: bool) -> torch.Tensor:
    if not stochastic:
        return mu
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


@torch.no_grad()
def analytic_semantics(
    energy: torch.Tensor,
    mask_prob: torch.Tensor,
    core_prob: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    energy_log_den: Optional[float] = None,
) -> torch.Tensor:
    """Compute stable 9D physical descriptors from soft wave-system masks.

    Parameters
    ----------
    energy: [B,1,H,W] or [B,H,W], non-negative display-domain energy.
    mask_prob: [B,K,H,W].
    core_prob: optional [B,K,H,W]. Core weighting sharpens peak estimation.
    """
    if energy.ndim == 3:
        energy = energy[:, None]
    B, K, H, W = mask_prob.shape
    dtype, device = energy.dtype, energy.device
    if energy_log_den is None:
        energy_log_den = math.log1p(H * W)

    E = torch.nan_to_num(energy, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0)
    M = torch.nan_to_num(mask_prob, nan=0.0).clamp(0, 1)
    part = E * M
    mass = part.sum(dim=(-2, -1)).clamp_min(eps)
    loge = torch.log1p(mass) / float(energy_log_den)

    fgrid = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, 1, H, 1)
    theta = torch.linspace(0, 2 * math.pi, W + 1, device=device, dtype=dtype)[:W]
    tgrid = theta.view(1, 1, 1, W)

    peak_src = part
    if core_prob is not None:
        core = torch.nan_to_num(core_prob, nan=0.0).clamp(0, 1)
        peak_src = part * (0.25 + 0.75 * core)

    # Smooth soft-peak estimator. It is more stable than hard argmax on noisy spectra.
    pnorm = peak_src / peak_src.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    peak_w = torch.softmax((12.0 * pnorm).flatten(2), dim=-1).view(B, K, H, W)
    fp = (peak_w * fgrid).sum(dim=(-2, -1))
    sx = (peak_w * torch.sin(tgrid)).sum(dim=(-2, -1))
    cx = (peak_w * torch.cos(tgrid)).sum(dim=(-2, -1))
    theta_p = torch.atan2(sx, cx)
    sinp = torch.sin(theta_p)
    cosp = torch.cos(theta_p)

    w = part / mass[:, :, None, None]
    df = fgrid - fp[:, :, None, None]
    dt = _wrap_angle(tgrid - theta_p[:, :, None, None]) / math.pi

    sf = torch.sqrt((w * df.square()).sum(dim=(-2, -1)).clamp_min(eps))
    st = torch.sqrt((w * dt.square()).sum(dim=(-2, -1)).clamp_min(eps))

    zf = df / sf[:, :, None, None].clamp_min(1e-3)
    zt = dt / st[:, :, None, None].clamp_min(1e-3)
    rho = (w * zf * zt).sum(dim=(-2, -1)).clamp(-0.95, 0.95)
    skew_f = (w * zf.pow(3)).sum(dim=(-2, -1)).clamp(-4.0, 4.0)
    skew_t = (w * zt.pow(3)).sum(dim=(-2, -1)).clamp(-4.0, 4.0)

    return torch.stack([
        loge, fp, sinp, cosp, sf, st, rho, skew_f, skew_t
    ], dim=-1)


class Canonicalizer(nn.Module):
    """Differentiably center/scale each slot in frequency-direction coordinates."""
    def __init__(self, cfg: V5AwareACSSVAEConfig):
        super().__init__()
        self.cfg = cfg
        u = torch.linspace(-cfg.canonical_f_extent, cfg.canonical_f_extent, cfg.canonical_h)
        v = torch.linspace(-cfg.canonical_t_extent, cfg.canonical_t_extent, cfg.canonical_w)
        self.register_buffer("u", u.view(1, 1, cfg.canonical_h, 1))
        self.register_buffer("v", v.view(1, 1, 1, cfg.canonical_w))

    def forward(
        self,
        energy: torch.Tensor,
        mask_prob: torch.Tensor,
        core_prob: torch.Tensor,
        support_prob: torch.Tensor,
        semantics: torch.Tensor,
    ) -> torch.Tensor:
        if energy.ndim == 3:
            energy = energy[:, None]
        B, K, H, W = mask_prob.shape
        eps = self.cfg.eps

        E = energy[:, None].expand(B, K, 1, H, W)
        M = mask_prob[:, :, None]
        C = core_prob[:, :, None]
        S = support_prob[:, :, None]
        part = E * M
        cpart = E * C
        spart = E * S

        peak = part.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        channels = torch.cat([
            part / peak,
            cpart / peak,
            spart / peak,
            M,
            C,
            S,
        ], dim=2).reshape(B * K, 6, H, W)

        sem = semantics.reshape(B * K, -1)
        fp = sem[:, 1].clamp(0, 1)
        theta = torch.atan2(sem[:, 2], sem[:, 3]) % (2 * math.pi)
        sf = sem[:, 4].clamp(self.cfg.min_spread_f, self.cfg.max_spread_f)
        st = sem[:, 5].clamp(self.cfg.min_spread_t, self.cfg.max_spread_t)

        uf = self.u.to(channels.dtype)
        vt = self.v.to(channels.dtype)
        f_sample = fp[:, None, None, None] + sf[:, None, None, None] * uf
        theta_sample = theta[:, None, None, None] + (math.pi * st[:, None, None, None]) * vt
        theta_frac = torch.remainder(theta_sample, 2 * math.pi) / (2 * math.pi)

        # Tile direction three times and sample from the middle copy.
        tiled = torch.cat([channels, channels, channels], dim=-1)
        y = 2.0 * f_sample - 1.0
        xpix = (1.0 + theta_frac) * W - 0.5
        x = 2.0 * xpix / max(3 * W - 1, 1) - 1.0
        grid = torch.stack([
            x.expand(-1, -1, self.cfg.canonical_h, self.cfg.canonical_w).squeeze(1),
            y.expand(-1, -1, self.cfg.canonical_h, self.cfg.canonical_w).squeeze(1),
        ], dim=-1)
        out = F.grid_sample(
            tiled,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out.reshape(B, K, 6, self.cfg.canonical_h, self.cfg.canonical_w)


class SlotEncoder(nn.Module):
    def __init__(self, cfg: V5AwareACSSVAEConfig):
        super().__init__()
        w = cfg.encoder_width
        self.image_net = nn.Sequential(
            ConvNormAct(6, w, 1),
            ResBlock(w),
            ConvNormAct(w, w * 2, 2),
            ResBlock(w * 2),
            ConvNormAct(w * 2, w * 3, 2),
            ResBlock(w * 3),
            ConvNormAct(w * 3, w * 4, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.context_net = nn.Sequential(
            nn.Linear(cfg.context_dim + cfg.semantic_dim + 1, cfg.hidden_dim // 2),
            nn.LayerNorm(cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim // 2),
            nn.GELU(),
        )
        image_dim = w * 4
        self.fuse = nn.Sequential(
            nn.Linear(image_dim + cfg.hidden_dim // 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.sem_mu = nn.Linear(cfg.hidden_dim, cfg.semantic_dim)
        self.sem_logvar = nn.Linear(cfg.hidden_dim, cfg.semantic_dim)
        self.shape_mu = nn.Linear(cfg.hidden_dim, cfg.shape_dim)
        self.shape_logvar = nn.Linear(cfg.hidden_dim, cfg.shape_dim)

        # Start from the analytic center and nearly-zero free shape code.
        nn.init.zeros_(self.sem_mu.weight)
        nn.init.zeros_(self.sem_mu.bias)
        nn.init.constant_(self.sem_logvar.bias, -4.0)
        nn.init.normal_(self.shape_mu.weight, std=0.01)
        nn.init.zeros_(self.shape_mu.bias)
        nn.init.constant_(self.shape_logvar.bias, -3.0)

    def forward(
        self,
        canonical: torch.Tensor,
        analytic_sem: torch.Tensor,
        context: torch.Tensor,
        exist_prob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, K = canonical.shape[:2]
        img = self.image_net(canonical.reshape(B * K, *canonical.shape[2:]))
        ctx_in = torch.cat([
            context.reshape(B * K, -1),
            analytic_sem.reshape(B * K, -1),
            exist_prob.reshape(B * K, 1),
        ], dim=-1)
        ctx = self.context_net(ctx_in)
        h = self.fuse(torch.cat([img, ctx], dim=-1))
        sem_mu = self.sem_mu(h).reshape(B, K, -1)
        sem_logvar = self.sem_logvar(h).clamp(-8.0, 3.0).reshape(B, K, -1)
        shape_mu = self.shape_mu(h).reshape(B, K, -1)
        shape_logvar = self.shape_logvar(h).clamp(-8.0, 3.0).reshape(B, K, -1)
        return {
            "sem_delta_mu": sem_mu,
            "sem_delta_logvar": sem_logvar,
            "shape_mu": shape_mu,
            "shape_logvar": shape_logvar,
        }


class SemanticRenderer(nn.Module):
    """Parameter-free generalized elliptical renderer from corrected semantics."""
    def __init__(self, cfg: V5AwareACSSVAEConfig):
        super().__init__()
        self.cfg = cfg
        f = torch.linspace(0, 1, cfg.n_freqs).view(1, 1, cfg.n_freqs, 1)
        t = torch.linspace(0, 2 * math.pi, cfg.n_dirs + 1)[:cfg.n_dirs].view(1, 1, 1, cfg.n_dirs)
        self.register_buffer("fgrid", f)
        self.register_buffer("tgrid", t)

    def forward(self, sem: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        eps = cfg.eps
        loge, fp = sem[..., 0], sem[..., 1]
        theta = torch.atan2(sem[..., 2], sem[..., 3])
        sf = sem[..., 4].clamp(cfg.min_spread_f, cfg.max_spread_f)
        st = sem[..., 5].clamp(cfg.min_spread_t, cfg.max_spread_t)
        rho = sem[..., 6].clamp(-cfg.max_abs_rho, cfg.max_abs_rho)
        skew_f = sem[..., 7].clamp(-cfg.max_abs_skew, cfg.max_abs_skew)
        skew_t = sem[..., 8].clamp(-cfg.max_abs_skew, cfg.max_abs_skew)

        df = (self.fgrid - fp[:, :, None, None]) / sf[:, :, None, None].clamp_min(1e-3)
        dt = _wrap_angle(self.tgrid - theta[:, :, None, None])
        dt = dt / (math.pi * st[:, :, None, None].clamp_min(1e-3))
        den = (1.0 - rho.square()).clamp_min(0.08)
        q = (df.square() + dt.square() - 2.0 * rho[:, :, None, None] * df * dt) / den[:, :, None, None]
        shape = torch.exp(-0.5 * q.clamp_max(60.0))
        asym = torch.exp(
            0.28 * skew_f[:, :, None, None] * torch.tanh(df)
            + 0.28 * skew_t[:, :, None, None] * torch.tanh(dt)
        ).clamp(0.15, 6.0)
        shape = shape * asym
        shape = shape / shape.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        mass = torch.expm1(loge * cfg.energy_log_den).clamp_min(0.0)
        return shape * mass[:, :, None, None]


class ShapeResidualNet(nn.Module):
    def __init__(self, cfg: V5AwareACSSVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.decoder_width
        self.fc = nn.Sequential(
            nn.Linear(cfg.semantic_dim + cfg.shape_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, w * 6 * 9),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            ConvNormAct(w, w, 1),
            ResBlock(w),
            nn.Upsample(size=(12, 18), mode="bilinear", align_corners=False),
            ConvNormAct(w, w, 1),
            ResBlock(w),
            nn.Upsample(size=(24, 36), mode="bilinear", align_corners=False),
            ConvNormAct(w, w // 2, 1),
            ResBlock(w // 2),
            nn.Upsample(size=(48, 72), mode="bilinear", align_corners=False),
            ConvNormAct(w // 2, w // 2, 1),
            nn.Conv2d(w // 2, 1, 3, padding=1),
        )

    def field(self, sem: torch.Tensor, zshape: torch.Tensor) -> torch.Tensor:
        B, K = sem.shape[:2]
        x = torch.cat([sem, zshape], dim=-1).reshape(B * K, -1)
        h = self.fc(x).view(B * K, self.cfg.decoder_width, 6, 9)
        out = self.net(h)[:, :, : self.cfg.n_freqs, : self.cfg.n_dirs]
        return out.reshape(B, K, self.cfg.n_freqs, self.cfg.n_dirs)

    def forward(self, sem: torch.Tensor, zshape: torch.Tensor) -> torch.Tensor:
        raw = self.field(sem, zshape)
        zero = self.field(sem, torch.zeros_like(zshape))
        return self.cfg.shape_log_residual_scale * torch.tanh(raw - zero)


class V5AwareStructuredSemanticVAE(nn.Module):
    def __init__(self, cfg: Optional[V5AwareACSSVAEConfig] = None):
        super().__init__()
        self.cfg = cfg or V5AwareACSSVAEConfig()
        self.canonicalizer = Canonicalizer(self.cfg)
        self.encoder = SlotEncoder(self.cfg)
        self.semantic_renderer = SemanticRenderer(self.cfg)
        self.shape_decoder = ShapeResidualNet(self.cfg)

    def apply_semantic_delta(self, base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        out = base.clone()
        out[..., 0] = (base[..., 0] + c.delta_loge * torch.tanh(delta[..., 0])).clamp_min(0.0)
        out[..., 1] = (base[..., 1] + c.delta_f * torch.tanh(delta[..., 1])).clamp(0.0, 1.0)

        theta0 = torch.atan2(base[..., 2], base[..., 3])
        # Use both sin/cos correction channels to form one stable tangent-angle correction.
        dtheta = c.delta_angle_rad * torch.tanh(0.7071 * (delta[..., 2] - delta[..., 3]))
        theta = theta0 + dtheta
        out[..., 2] = torch.sin(theta)
        out[..., 3] = torch.cos(theta)

        out[..., 4] = (
            base[..., 4].clamp_min(c.min_spread_f)
            * torch.exp(c.delta_log_spread * torch.tanh(delta[..., 4]))
        ).clamp(c.min_spread_f, c.max_spread_f)
        out[..., 5] = (
            base[..., 5].clamp_min(c.min_spread_t)
            * torch.exp(c.delta_log_spread * torch.tanh(delta[..., 5]))
        ).clamp(c.min_spread_t, c.max_spread_t)
        out[..., 6] = torch.tanh(
            _atanh_safe(base[..., 6], c.eps)
            + c.delta_rho_logit * torch.tanh(delta[..., 6])
        ).clamp(-c.max_abs_rho, c.max_abs_rho)
        out[..., 7] = (
            base[..., 7] + c.delta_skew * torch.tanh(delta[..., 7])
        ).clamp(-c.max_abs_skew, c.max_abs_skew)
        out[..., 8] = (
            base[..., 8] + c.delta_skew * torch.tanh(delta[..., 8])
        ).clamp(-c.max_abs_skew, c.max_abs_skew)
        return out

    def decode_from_codes(
        self,
        semantics: torch.Tensor,
        shape_code: torch.Tensor,
        exist_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base = self.semantic_renderer(semantics)
        residual_log = self.shape_decoder(semantics, shape_code)
        # Multiplicative residual in log domain, followed by exact energy renormalization.
        part_hat = (base + self.cfg.eps) * torch.exp(residual_log)
        target_mass = torch.expm1(semantics[..., 0] * self.cfg.energy_log_den).clamp_min(0.0)
        part_hat = part_hat * (
            target_mass[:, :, None, None]
            / part_hat.sum(dim=(-2, -1), keepdim=True).clamp_min(self.cfg.eps)
        )
        if exist_prob is not None:
            part_hat = part_hat * exist_prob[:, :, None, None]
            base = base * exist_prob[:, :, None, None]
        return {
            "semantic_base": base,
            "shape_residual_log": residual_log,
            "part_hat": part_hat,
        }

    def forward(
        self,
        energy: torch.Tensor,
        mask_prob: torch.Tensor,
        core_prob: torch.Tensor,
        support_prob: torch.Tensor,
        exist_prob: torch.Tensor,
        slot_context: torch.Tensor,
        stochastic: bool = True,
        shape_code_override: Optional[torch.Tensor] = None,
        semantic_delta_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        analytic = analytic_semantics(
            energy,
            mask_prob,
            core_prob=core_prob,
            eps=self.cfg.eps,
            energy_log_den=self.cfg.energy_log_den,
        )
        canonical = self.canonicalizer(
            energy,
            mask_prob,
            core_prob,
            support_prob,
            analytic,
        )
        enc = self.encoder(canonical, analytic, slot_context, exist_prob)
        sem_delta = (
            semantic_delta_override
            if semantic_delta_override is not None
            else reparameterize(enc["sem_delta_mu"], enc["sem_delta_logvar"], stochastic)
        )
        zshape = (
            shape_code_override
            if shape_code_override is not None
            else reparameterize(enc["shape_mu"], enc["shape_logvar"], stochastic)
        )
        corrected = self.apply_semantic_delta(analytic, sem_delta)
        dec = self.decode_from_codes(corrected, zshape, exist_prob=exist_prob)
        return {
            "analytic_semantics": analytic,
            "corrected_semantics": corrected,
            "semantic_delta": sem_delta,
            "shape_code": zshape,
            "canonical": canonical,
            **enc,
            **dec,
        }

    @property
    def token_dim_per_wave(self) -> int:
        return self.cfg.semantic_dim + self.cfg.shape_dim

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)
    if weight is None:
        return kl.mean()
    w = weight.float()
    return (kl * w).sum() / w.sum().clamp_min(1.0)


def covariance_penalty(x: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Off-diagonal covariance penalty for [B,K,D] latent tensors."""
    D = x.shape[-1]
    flat = x.reshape(-1, D)
    if weight is not None:
        w = weight.reshape(-1).float()
        keep = w > 0.25
        flat = flat[keep]
    if flat.shape[0] < max(4, D):
        return x.new_zeros(())
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.T @ flat / max(flat.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.square().mean()


def semantic_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-slot interpretable semantic distance with circular direction handling."""
    d = []
    d.append((pred[..., 0] - target[..., 0]).abs())
    d.append(2.0 * (pred[..., 1] - target[..., 1]).abs())
    tp = torch.atan2(pred[..., 2], pred[..., 3])
    tt = torch.atan2(target[..., 2], target[..., 3])
    d.append(_wrap_angle(tp - tt).abs() / math.pi)
    d.append(2.0 * (pred[..., 4] - target[..., 4]).abs())
    d.append(1.5 * (pred[..., 5] - target[..., 5]).abs())
    d.append(0.5 * (pred[..., 6] - target[..., 6]).abs())
    d.append(0.15 * (pred[..., 7] - target[..., 7]).abs())
    d.append(0.15 * (pred[..., 8] - target[..., 8]).abs())
    return torch.stack(d, dim=-1).mean(dim=-1)


def marginal_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    p = pred.clamp_min(0)
    t = target.clamp_min(0)
    pf = p.sum(dim=-1); tf = t.sum(dim=-1)
    pt = p.sum(dim=-2); tt = t.sum(dim=-2)
    pf = pf / pf.sum(dim=-1, keepdim=True).clamp_min(eps)
    tf = tf / tf.sum(dim=-1, keepdim=True).clamp_min(eps)
    pt = pt / pt.sum(dim=-1, keepdim=True).clamp_min(eps)
    tt = tt / tt.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (pf - tf).abs().mean(), (pt - tt).abs().mean()


def reconstruction_terms(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    w = weight[:, :, None, None].float()
    denom = w.sum().clamp_min(1.0)
    mse_slot = (pred - target).square().mean(dim=(-2, -1))
    log_slot = (
        torch.log1p(20.0 * pred.clamp_min(0))
        - torch.log1p(20.0 * target.clamp_min(0))
    ).abs().mean(dim=(-2, -1))
    mass_p = pred.sum(dim=(-2, -1))
    mass_t = target.sum(dim=(-2, -1))
    energy_slot = (mass_p - mass_t).abs() / mass_t.clamp_min(eps)
    mse = (mse_slot * weight).sum() / denom
    log_l1 = (log_slot * weight).sum() / denom
    energy = (energy_slot * weight).sum() / denom
    freq, direction = marginal_l1(pred * w, target * w, eps=eps)
    return {
        "mse": mse,
        "log_l1": log_l1,
        "energy": energy,
        "freq": freq,
        "direction": direction,
    }


__all__ = [
    "V5AwareACSSVAEConfig",
    "V5AwareStructuredSemanticVAE",
    "analytic_semantics",
    "kl_standard_normal",
    "covariance_penalty",
    "semantic_distance",
    "reconstruction_terms",
    "marginal_l1",
]

# =============================================================================
# ICWDS Stage-2 hybrid codec components (C3-D / C3-F / C4-B)
# =============================================================================

NF = 47
ND = 72
K_MAX = 6

class PrototypeSelector(nn.Module):
    def __init__(self, input_dim, n_codes, width=192, dropout=0.10):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_codes)

    def forward(self, x):
        x = self.norm(x)
        h = F.gelu(self.fc1(x))
        r = h
        h = self.drop(F.gelu(self.fc2(h)))
        h = F.gelu(self.fc3(h) + r)
        return self.head(h)

class PeriodicConv2d(nn.Module):
    def __init__(self, ci, co, kernel=3, stride=1, bias=False):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(ci, co, kernel_size=kernel, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        p = self.pad
        x = F.pad(x, (p, p, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, p, p), mode="replicate")
        return self.conv(x)

class PResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = PeriodicConv2d(ch, ch, 3, 1, False)
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.c2 = PeriodicConv2d(ch, ch, 3, 1, False)
        self.n2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(x + h)

class NestedBitEncoder(nn.Module):
    def __init__(self, max_bits, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            PeriodicConv2d(4, w, 3, 2, False),
            nn.GroupNorm(min(8, w), w), nn.GELU(), PResBlock(w),
            PeriodicConv2d(w, w*2, 3, 2, False),
            nn.GroupNorm(min(8, w*2), w*2), nn.GELU(), PResBlock(w*2),
            PeriodicConv2d(w*2, w*3, 3, 2, False),
            nn.GroupNorm(min(8, w*3), w*3), nn.GELU(), PResBlock(w*3),
        )
        ch = w * 3
        self.attn = nn.Conv2d(ch, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(ch * 3, ch * 2),
            nn.LayerNorm(ch * 2),
            nn.GELU(),
            nn.Linear(ch * 2, max_bits),
        )

    def forward(self, residual_target, prior, gate):
        x = torch.stack([residual_target, residual_target.abs(), prior, gate], dim=1)
        h = self.stem(x)
        flat = h.flatten(2)
        attn = torch.softmax(self.attn(h).flatten(2), dim=-1)
        ap = (flat * attn).sum(dim=-1)
        mx = F.adaptive_max_pool2d(h, 1).flatten(1)
        av = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(torch.cat([ap, mx, av], dim=-1))

def ste_binary(logits):
    soft = torch.tanh(logits)
    hard = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
    q = soft + (hard - soft).detach()
    return q, soft

def prefix_mask(n_bits, max_bits, device, dtype):
    if int(n_bits) < 0 or int(n_bits) > int(max_bits):
        raise ValueError((n_bits, max_bits))
    return (torch.arange(max_bits, device=device) < int(n_bits)).to(dtype=dtype)[None]

class CondPyramid(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.c0 = nn.Sequential(
            PeriodicConv2d(2, w, 3, 1, False),
            nn.GroupNorm(min(8, w), w), nn.GELU(),
        )
        self.c1 = nn.Sequential(
            PeriodicConv2d(w, w*2, 3, 2, False),
            nn.GroupNorm(min(8, w*2), w*2), nn.GELU(),
        )
        self.c2 = nn.Sequential(
            PeriodicConv2d(w*2, w*3, 3, 2, False),
            nn.GroupNorm(min(8, w*3), w*3), nn.GELU(),
        )
        self.c3 = nn.Sequential(
            PeriodicConv2d(w*3, w*3, 3, 2, False),
            nn.GroupNorm(min(8, w*3), w*3), nn.GELU(),
        )

    def forward(self, prior, gate):
        x = torch.stack([prior, gate], dim=1)
        c0 = self.c0(x)
        c1 = self.c1(c0)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        return c0, c1, c2, c3

class CodeFiLM(nn.Module):
    """Near-identity code injection. Final layer is zero-initialized."""
    def __init__(self, max_bits, channels):
        super().__init__()
        hid = max(48, channels)
        self.net = nn.Sequential(
            nn.Linear(max_bits + 1, hid),
            nn.GELU(),
            nn.Linear(hid, channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, qbits, bit_fraction):
        B = qbits.shape[0]
        bf = qbits.new_full((B, 1), float(bit_fraction))
        gamma, beta = self.net(torch.cat([qbits, bf], dim=-1)).chunk(2, dim=-1)
        gamma = 0.20 * torch.tanh(gamma)[:, :, None, None]
        beta = 0.20 * torch.tanh(beta)[:, :, None, None]
        return h * (1.0 + gamma) + beta

class DetailAwareResidualDecoder(nn.Module):
    def __init__(self, max_bits, width=32, branch="intra"):
        super().__init__()
        self.max_bits = int(max_bits)
        self.branch = str(branch)
        w = width
        self.cond = CondPyramid(w)
        self.bit_fc = nn.Sequential(nn.Linear(max_bits, w*3*6*9), nn.GELU())
        self.b3 = PResBlock(w*3)
        self.p2 = nn.Conv2d(w*3, w*3, 1)
        self.b2 = PResBlock(w*3)
        self.p1 = nn.Conv2d(w*3, w*2, 1)
        self.b1 = PResBlock(w*2)
        self.p0 = nn.Conv2d(w*2, w, 1)
        self.b0 = PResBlock(w)
        self.out = PeriodicConv2d(w, 1, 3, 1, True)
        self.scale = nn.Parameter(torch.tensor(0.35))

        # New C3-F modules.
        self.film3 = CodeFiLM(max_bits, w*3)
        self.film2 = CodeFiLM(max_bits, w*3)
        self.film1 = CodeFiLM(max_bits, w*2)
        self.film0 = CodeFiLM(max_bits, w)
        self.detail_block = PResBlock(w)
        self.detail_out = PeriodicConv2d(w, 1, 3, 1, True)
        nn.init.zeros_(self.detail_out.conv.weight)
        if self.detail_out.conv.bias is not None:
            nn.init.zeros_(self.detail_out.conv.bias)
        self.detail_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, qbits, prior, gate, n_bits=None):
        B = prior.shape[0]
        if n_bits is None:
            n_bits = int((qbits.abs().sum(dim=0) > 0).sum().item())
        frac = float(n_bits) / float(self.max_bits)
        c0, c1, c2, c3 = self.cond(prior, gate)

        h = self.bit_fc(qbits).view(B, c3.shape[1], 6, 9)
        h = self.b3(h + c3)
        h = self.film3(h, qbits, frac)

        h = F.interpolate(h, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.b2(self.p2(h) + c2)
        h = self.film2(h, qbits, frac)

        h = F.interpolate(h, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.b1(self.p1(h) + c1)
        h = self.film1(h, qbits, frac)

        h = F.interpolate(h, size=(48, 72), mode="bilinear", align_corners=False)
        c0p = F.pad(c0, (0, 0, 0, 1), mode="replicate")
        h = self.b0(self.p0(h) + c0p)
        h = self.film0(h, qbits, frac)

        base = self.out(h)[:, 0, :NF, :ND]
        detail = self.detail_out(self.detail_block(h))[:, 0, :NF, :ND]
        # High-pass detail residual; direction is periodic, frequency is replicated.
        dp = F.pad(detail[:, None], (1, 1, 0, 0), mode="circular")
        dp = F.pad(dp, (0, 0, 1, 1), mode="replicate")
        blur = F.avg_pool2d(dp, 3, stride=1)[:, 0]
        detail_hp = detail - blur
        raw = torch.tanh(base + torch.sigmoid(self.detail_scale) * detail_hp)

        amp = 0.65 * torch.sigmoid(self.scale)
        spatial = (0.08 + 0.92 * gate) if self.branch == "intra" else (1.0 - 0.72 * gate)
        return amp * raw * spatial

class AdaptiveDualResidualCodec(nn.Module):
    def __init__(self, max_bits=24, enc_width=32, dec_width=32):
        super().__init__()
        self.max_bits = int(max_bits)
        self.intra_encoder = NestedBitEncoder(max_bits, enc_width)
        self.bg_encoder = NestedBitEncoder(max_bits, enc_width)
        self.intra_decoder = DetailAwareResidualDecoder(max_bits, dec_width, branch="intra")
        self.bg_decoder = DetailAwareResidualDecoder(max_bits, dec_width, branch="bg")

    def encode(self, prior, gate, intra_target, bg_target):
        li = self.intra_encoder(intra_target, prior, gate)
        lb = self.bg_encoder(bg_target, prior, gate)
        return li, lb

    def decode_logits(self, logits, n_bits, prior, gate, branch, q_override=None):
        if int(n_bits) == 0:
            return torch.zeros_like(prior), None, None
        q, soft = ste_binary(logits)
        mask = prefix_mask(n_bits, self.max_bits, logits.device, logits.dtype)
        q = q * mask
        if q_override is not None:
            q = q_override * mask
        dec = self.intra_decoder if branch == "intra" else self.bg_decoder
        pred = dec(q, prior, gate, n_bits=int(n_bits))
        return pred, q, soft

    def decode_pair(self, li, lb, prior, gate, intra_bits, bg_bits):
        ri, qi, si = self.decode_logits(li, intra_bits, prior, gate, "intra")
        rb, qb, sb = self.decode_logits(lb, bg_bits, prior, gate, "bg")
        return {"ri": ri, "rb": rb, "qi": qi, "qb": qb, "si": si, "sb": sb}

    def forward(self, prior, gate, intra_target, bg_target, intra_bits, bg_bits):
        li, lb = self.encode(prior, gate, intra_target, bg_target)
        out = self.decode_pair(li, lb, prior, gate, intra_bits, bg_bits)
        out.update({"li": li, "lb": lb})
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class RateConditionedController(nn.Module):
    def __init__(self, input_dim, n_alloc, width=160, dropout=0.08):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim + 1)
        self.fc1 = nn.Linear(input_dim + 1, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_alloc)

    def forward(self, x, lambda_norm):
        if lambda_norm.ndim == 1:
            lambda_norm = lambda_norm[:, None]
        h = self.norm(torch.cat([x, lambda_norm], dim=-1))
        h = F.gelu(self.fc1(h)); r = h
        h = self.drop(F.gelu(self.fc2(h)))
        h = F.gelu(self.fc3(h) + r)
        return self.head(h)

def branch_scalar_features(x):
    ax = x.abs()
    df = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(-2, -1))
    dd = (torch.roll(x, -1, -1) - x).abs().mean(dim=(-2, -1))
    lap = torch.zeros_like(x)
    lap[:, 1:-1] = x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]
    return torch.stack([
        ax.mean(dim=(-2, -1)), ax.std(dim=(-2, -1)), ax.amax(dim=(-2, -1)),
        df, dd, lap.abs().mean(dim=(-2, -1)),
    ], dim=-1)

def controller_features(batch, li, lb):
    intra = branch_scalar_features(batch["intra_target"])
    bg = branch_scalar_features(batch["bg_struct"])
    prior = branch_scalar_features(batch["main_prior"])
    gate = torch.stack([
        batch["prior_gate"].mean(dim=(-2, -1)),
        batch["prior_gate"].std(dim=(-2, -1)),
    ], dim=-1)
    conf_i = torch.tanh(li).abs(); conf_b = torch.tanh(lb).abs()
    prefix_points = [4, 8, 12, 16]
    ci = torch.stack([conf_i[:, :p].mean(-1) for p in prefix_points], dim=-1)
    cb = torch.stack([conf_b[:, :p].mean(-1) for p in prefix_points], dim=-1)
    ratio = torch.stack([
        intra[:, 0] / prior[:, 0].clamp_min(1e-5),
        bg[:, 0] / prior[:, 0].clamp_min(1e-5),
    ], dim=-1)
    return torch.cat([intra, bg, prior, gate, ci, cb, ratio], dim=-1)


class ProtocolRuntime:
    """Runtime view of the frozen C4-A/C4-B exact 30-byte packet protocol."""

    def __init__(self, protocol: dict):
        self.protocol = protocol
        self.semantic_fields = list(protocol["semantic_fields"])
        self.field_specs = list(protocol["field_specs"])
        self.allocation_menu = [tuple(map(int, a)) for a in protocol["reduced_allocation_menu"]]
        self.morphology_k = int(protocol["morphology_K"])
        self.morphology_bits = int(protocol["morphology_bits_per_wave"])
        self.packet_bytes = int(protocol["packet_bytes"])
        self.payload_bytes = int(protocol["payload_bytes"])
        self.crc_bytes = int(protocol["crc_bytes"])
        self.semantic_profiles = {}
        for key, bits in protocol["semantic_profiles"].items():
            left, right = key.split("_")
            n = int(left[1:])
            aid = int(right[1:])
            self.semantic_profiles[(n, aid)] = list(map(int, bits))
        if self.packet_bytes != 30:
            raise ValueError(f"Expected exact 30-byte packet protocol, got {self.packet_bytes}")

    @staticmethod
    def sem_internal_to_packet(sem9: torch.Tensor) -> torch.Tensor:
        theta = torch.remainder(torch.atan2(sem9[..., 2], sem9[..., 3]), 2 * math.pi) / (2 * math.pi)
        return torch.stack([
            sem9[..., 0], sem9[..., 1], theta, sem9[..., 4], sem9[..., 5],
            sem9[..., 6], sem9[..., 7], sem9[..., 8],
        ], dim=-1)

    @staticmethod
    def sem_packet_to_internal(p8: torch.Tensor) -> torch.Tensor:
        theta = p8[..., 2] * (2 * math.pi)
        return torch.stack([
            p8[..., 0], p8[..., 1], torch.sin(theta), torch.cos(theta),
            p8[..., 3], p8[..., 4], p8[..., 5], p8[..., 6], p8[..., 7],
        ], dim=-1)

    @staticmethod
    def _forward_compand(x: torch.Tensor, spec: dict) -> torch.Tensor:
        kind = spec["kind"]
        lo, hi = float(spec["lo"]), float(spec["hi"])
        if kind == "circular":
            return torch.remainder((x - lo) / (hi - lo), 1.0)
        if kind == "linear":
            return ((x - lo) / (hi - lo)).clamp(0, 1)
        if kind == "log":
            lx = torch.log(x.clamp_min(lo))
            l0, l1 = math.log(lo), math.log(hi)
            return ((lx - l0) / (l1 - l0)).clamp(0, 1)
        if kind == "asinh":
            s = float(spec.get("scale", 1.0))
            a0, a1 = math.asinh(lo / s), math.asinh(hi / s)
            return ((torch.asinh(x / s) - a0) / (a1 - a0)).clamp(0, 1)
        raise ValueError(kind)

    @staticmethod
    def _inverse_compand(y: torch.Tensor, spec: dict) -> torch.Tensor:
        kind = spec["kind"]
        lo, hi = float(spec["lo"]), float(spec["hi"])
        if kind in ("linear", "circular"):
            x = lo + (hi - lo) * y
            if kind == "circular":
                x = lo + torch.remainder(x - lo, hi - lo)
            return x
        if kind == "log":
            l0, l1 = math.log(lo), math.log(hi)
            return torch.exp(l0 + (l1 - l0) * y)
        if kind == "asinh":
            s = float(spec.get("scale", 1.0))
            a0, a1 = math.asinh(lo / s), math.asinh(hi / s)
            return s * torch.sinh(a0 + (a1 - a0) * y)
        raise ValueError(kind)

    def packet_fields_to_unit(self, p8: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self._forward_compand(p8[..., d], self.field_specs[d]) for d in range(8)
        ], dim=-1)

    def unit_to_packet_fields(self, y8: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self._inverse_compand(y8[..., d], self.field_specs[d]) for d in range(8)
        ], dim=-1)

    @staticmethod
    def project_unit_fields(y8: torch.Tensor) -> torch.Tensor:
        cols = []
        for d in range(8):
            v = y8[..., d]
            cols.append(torch.remainder(v, 1.0) if d == 2 else v.clamp(0, 1))
        return torch.stack(cols, dim=-1)

    def profile_bits_tensor(self, counts: torch.Tensor, aids: torch.Tensor, device) -> torch.Tensor:
        rows = []
        for n, a in zip(counts.detach().cpu().tolist(), aids.detach().cpu().tolist()):
            rows.append(self.semantic_profiles[(max(1, int(n)), int(a))])
        return torch.tensor(rows, dtype=torch.long, device=device)


class PacketSemanticQATAdapter(nn.Module):
    """C4-B bounded packet-domain semantic pre-quantization adapter."""

    def __init__(self, width: int = 96, delta_steps: float = 1.25, protocol: ProtocolRuntime | None = None):
        super().__init__()
        self.delta_steps = float(delta_steps)
        self.protocol = protocol
        self.net = nn.Sequential(
            nn.LayerNorm(21),
            nn.Linear(21, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 8),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def set_protocol(self, protocol: ProtocolRuntime) -> None:
        self.protocol = protocol

    def _proto(self) -> ProtocolRuntime:
        if self.protocol is None:
            raise RuntimeError("PacketSemanticQATAdapter requires a ProtocolRuntime")
        return self.protocol

    def adjust_unit(self, sem9, counts, aids, active):
        proto = self._proto()
        p8 = proto.sem_internal_to_packet(sem9)
        y = proto.packet_fields_to_unit(p8)
        bits = proto.profile_bits_tensor(counts, aids, sem9.device)
        levels = (torch.pow(2.0, bits.float()) - 1.0).clamp_min(1.0)
        step = 1.0 / levels
        bit_ctx = (bits.float() / 9.0)[:, None, :].expand(-1, sem9.shape[1], -1)
        cnt_ctx = (counts.float() / float(K_MAX))[:, None, None].expand(-1, sem9.shape[1], 1)
        aid_ctx = F.one_hot(aids.long(), num_classes=4).float()[:, None, :].expand(-1, sem9.shape[1], -1)
        x = torch.cat([y, bit_ctx, cnt_ctx, aid_ctx], dim=-1)
        raw = self.net(x)
        delta = self.delta_steps * step[:, None, :] * torch.tanh(raw)
        y_adj = proto.project_unit_fields(y + delta * active[..., None])
        return y, y_adj, step, bits

    def fake_quantize(self, sem9, counts, aids, active):
        proto = self._proto()
        y, y_adj, step, bits = self.adjust_unit(sem9, counts, aids, active)
        levels = (torch.pow(2.0, bits.float()) - 1.0).clamp_min(1.0)[:, None, :]
        hard = torch.round(y_adj * levels) / levels
        yq = y_adj + (hard - y_adj).detach()
        semq = proto.sem_packet_to_internal(proto.unit_to_packet_fields(yq))
        return semq, dict(y=y, y_adj=y_adj, step=step, bits=bits, hard=hard)

    @torch.no_grad()
    def hard_quantize(self, sem9, counts, aids, active):
        proto = self._proto()
        y, y_adj, step, bits = self.adjust_unit(sem9, counts, aids, active)
        levels = (torch.pow(2.0, bits.float()) - 1.0).clamp_min(1.0)[:, None, :]
        q = torch.round(y_adj * levels).long().clamp_min(0)
        hard = q.float() / levels
        semq = proto.sem_packet_to_internal(proto.unit_to_packet_fields(hard))
        return q, semq, dict(y=y, y_adj=y_adj, step=step, bits=bits, hard=hard)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class PacketAwareController(nn.Module):
    def __init__(self,input_dim,width=160,dropout=0.08,n_alloc=4):
        super().__init__(); self.norm=nn.LayerNorm(input_dim+1)
        self.fc1=nn.Linear(input_dim+1,width); self.fc2=nn.Linear(width,width); self.fc3=nn.Linear(width,width)
        self.drop=nn.Dropout(dropout); self.head=nn.Linear(width,n_alloc)
    def forward(self,x,lambda_norm):
        if lambda_norm.ndim==1: lambda_norm=lambda_norm[:,None]
        h=self.norm(torch.cat([x,lambda_norm],dim=-1)); h=F.gelu(self.fc1(h)); r=h
        h=self.drop(F.gelu(self.fc2(h))); h=F.gelu(self.fc3(h)+r); return self.head(h)


class ICWDSFrontEndCodec(nn.Module):
    """
    Consolidated Stage-2 front-end feature codec.

    Historical project naming keeps the archive name ``VAE.pt`` for compatibility,
    but the deployed object is a hybrid semantic/morphology/residual packet codec:
      C2 AC-SSVAE backbone + C3-D morphology selector + C3-F dual residual codec
      + C4-B packet QAT adapter + packet-aware allocation controller.
    """

    def __init__(
        self,
        base_model: V5AwareStructuredSemanticVAE,
        morph_selector: PrototypeSelector,
        residual_codec: AdaptiveDualResidualCodec,
        qat_adapter: PacketSemanticQATAdapter,
        packet_controller: PacketAwareController,
        protocol_runtime: ProtocolRuntime,
        morph_centers: torch.Tensor,
        selector_x_mean: torch.Tensor,
        selector_x_std: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        lambdas: list[float],
    ):
        super().__init__()
        self.base_model = base_model
        self.morph_selector = morph_selector
        self.residual_codec = residual_codec
        self.qat_adapter = qat_adapter
        self.packet_controller = packet_controller
        self.protocol_runtime = protocol_runtime
        self.qat_adapter.set_protocol(protocol_runtime)
        self.register_buffer("morph_centers", morph_centers.float())
        self.register_buffer("selector_x_mean", selector_x_mean.float())
        self.register_buffer("selector_x_std", selector_x_std.float().clamp_min(1e-5))
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1e-5))
        self.lambdas = [float(x) for x in lambdas]

    @torch.no_grad()
    def decode_semantic_prior(self, sem9, morph_idx, active):
        z = self.morph_centers.to(sem9.device)[morph_idx.long()]
        dec = self.base_model.decode_from_codes(sem9, z, exist_prob=active)
        return dec["part_hat"].sum(1).clamp(0, 1)

    def quantize_semantics(self, sem9, counts, allocation_ids, active, hard=True):
        if hard:
            return self.qat_adapter.hard_quantize(sem9, counts, allocation_ids, active)
        return self.qat_adapter.fake_quantize(sem9, counts, allocation_ids, active)

    @torch.no_grad()
    def choose_allocation(self, features: torch.Tensor, mode: str = "quality") -> torch.Tensor:
        if mode not in {"quality", "comm"}:
            raise ValueError("mode must be 'quality' or 'comm'")
        lam = self.lambdas[0] if mode == "quality" else self.lambdas[-1]
        max_lam = max(self.lambdas)
        x = (features - self.feature_mean) / self.feature_std
        ln = x.new_full((len(x), 1), lam / max(max_lam, 1e-12))
        return self.packet_controller(x, ln).argmax(-1)

    @classmethod
    def from_archive(cls, archive_path: str, map_location: str | torch.device = "cpu"):
        bundle = torch.load(archive_path, map_location=map_location, weights_only=False)
        if bundle.get("format") != "ICWDS_VAE_ARCHIVE_V1":
            raise ValueError("Not an ICWDS consolidated VAE.pt archive")

        c2 = bundle["c2_checkpoint"]
        c3d = bundle["c3d_package"]
        c3f = bundle["c3f_package"]
        c4b = bundle["c4b_package"]

        cfg_dict = c2.get("config", {})
        allowed = set(V5AwareACSSVAEConfig.__dataclass_fields__)
        cfg = V5AwareACSSVAEConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})
        base_model = V5AwareStructuredSemanticVAE(cfg)
        base_model.load_state_dict(c2["model"], strict=True)

        morph_k = int(c4b.get("protocol", {}).get("morphology_K", 4))
        sel_item = c3d["selectors"][morph_k]
        morph_selector = PrototypeSelector(
            input_dim=int(c3d["input_dim"]),
            n_codes=morph_k,
            width=int(sel_item.get("width", 192)),
            dropout=float(sel_item.get("dropout", 0.10)),
        )
        morph_selector.load_state_dict(sel_item["state_dict"], strict=True)

        c3f_ctrl = c3f.get("control", {})
        residual_codec = AdaptiveDualResidualCodec(
            max_bits=int(c3f_ctrl.get("MAX_BITS_PER_BRANCH", 24)),
            enc_width=int(c3f_ctrl.get("ENC_WIDTH", 32)),
            dec_width=int(c3f_ctrl.get("DEC_WIDTH", 32)),
        )
        residual_codec.load_state_dict(c3f["codec"], strict=True)

        proto = ProtocolRuntime(c4b["protocol"])
        qa = c4b["qat_adapter_arch"]
        qat_adapter = PacketSemanticQATAdapter(
            width=int(qa["width"]),
            delta_steps=float(qa["delta_steps"]),
            protocol=proto,
        )
        qat_adapter.load_state_dict(c4b["qat_adapter"], strict=True)

        pa = c4b["packet_controller_arch"]
        packet_controller = PacketAwareController(
            input_dim=int(pa["input_dim"]),
            width=int(pa["width"]),
            dropout=float(pa["dropout"]),
            n_alloc=int(pa["n_alloc"]),
        )
        packet_controller.load_state_dict(c4b["packet_controller"], strict=True)

        model = cls(
            base_model=base_model,
            morph_selector=morph_selector,
            residual_codec=residual_codec,
            qat_adapter=qat_adapter,
            packet_controller=packet_controller,
            protocol_runtime=proto,
            morph_centers=sel_item["centers"],
            selector_x_mean=c3d["x_mean"],
            selector_x_std=c3d["x_std"],
            feature_mean=c4b["feature_mean"],
            feature_std=c4b["feature_std"],
            lambdas=list(c4b["lambdas"]),
        )
        return model


def build_vae_archive_from_files(
    c2_checkpoint_path: str,
    c3d_package_path: str,
    c3f_package_path: str,
    c4b_package_path: str,
    output_path: str = "VAE.pt",
) -> str:
    """Consolidate the four validated Stage-2 artifacts into one portable VAE.pt."""
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    bundle = {
        "format": "ICWDS_VAE_ARCHIVE_V1",
        "description": "C2 AC-SSVAE + C3-D morphology + C3-F residual codec + C4-B QAT/controller",
        "c2_checkpoint": load(c2_checkpoint_path),
        "c3d_package": load(c3d_package_path),
        "c3f_package": load(c3f_package_path),
        "c4b_package": load(c4b_package_path),
    }
    torch.save(bundle, output_path)
    return output_path


def build_vae_archive_from_hf(
    output_path: str = "VAE.pt",
    repo_id: str = "wuff-mann/Wave_compress_system",
    token: str | None = None,
) -> str:
    """Download the four frozen artifacts from HF and build one self-contained VAE.pt."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first") from exc

    paths = {
        "c2": "WaveSemanticHybridCodec/V5_ACSSVAE_C2_CausalProtectedReal/checkpoints/C2_consolidate/complete.pt",
        "c3d": "WaveSemanticHybridCodec/V5_ACSSVAE_C3D_ReconstructionAwarePrototypeSelector/C3D_prototype_selector_package.pt",
        "c3f": "WaveSemanticHybridCodec/V5_ACSSVAE_C3F_AdaptiveDualResidualCodec/C3F_adaptive_dual_residual_codec_package.pt",
        "c4b": "WaveSemanticHybridCodec/V5_ACSSVAE_C4B_PacketAwareQAT_Controller/C4B_packet_qat_controller_package.pt",
    }
    local = {
        k: hf_hub_download(repo_id=repo_id, filename=v, repo_type="model", token=token)
        for k, v in paths.items()
    }
    return build_vae_archive_from_files(local["c2"], local["c3d"], local["c3f"], local["c4b"], output_path)


def load_vae_archive(path: str = "VAE.pt", map_location: str | torch.device = "cpu") -> ICWDSFrontEndCodec:
    return ICWDSFrontEndCodec.from_archive(path, map_location=map_location)


__all__ = [
    "V5AwareACSSVAEConfig",
    "V5AwareStructuredSemanticVAE",
    "PrototypeSelector",
    "AdaptiveDualResidualCodec",
    "RateConditionedController",
    "PacketSemanticQATAdapter",
    "PacketAwareController",
    "ProtocolRuntime",
    "ICWDSFrontEndCodec",
    "build_vae_archive_from_files",
    "build_vae_archive_from_hf",
    "load_vae_archive",
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build consolidated ICWDS VAE.pt archive")
    parser.add_argument("--build-from-hf", action="store_true")
    parser.add_argument("--output", default="VAE.pt")
    parser.add_argument("--repo", default="wuff-mann/Wave_compress_system")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    if args.build_from_hf:
        out = build_vae_archive_from_hf(args.output, args.repo, args.token)
        print(out)
