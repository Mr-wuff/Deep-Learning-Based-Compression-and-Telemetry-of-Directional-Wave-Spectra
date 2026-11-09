#!/usr/bin/env python3
"""Download original HF research files listed in data/artifact_manifest.json.

Safe by default: reports only, no execution or deserialization of checkpoints.
Uses the Python standard library; optionally HF_TOKEN for permitted private assets.
Records hashes and repository revision in a reproducibility lock file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "artifact_manifest.json"
GROUPS = ("reports", "code", "models")


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def api_revision(repo_id: str, repo_type: str, token: str | None) -> str | None:
    api_kind = "datasets" if repo_type == "dataset" else "models"
    url = "https://huggingface.co/api/" + api_kind + "/" + repo_id
    headers = {"User-Agent": "ICWDS-reproducibility/1.0"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as res:
            data = json.load(res)
        return data.get("sha")
    except (urllib.error.URLError, ValueError, OSError):
        return None


def download_file(item: dict, root: pathlib.Path, revision: str, token: str | None):
    repo = item["repo_id"]
    src_path = item["path"]
    filename_parts = [urllib.parse.quote(part, safe="") for part in src_path.split("/")]
    url = (
        "https://huggingface.co/" + ("datasets/" if item["repo_type"] == "dataset" else "")
        + repo + "/resolve/" + urllib.parse.quote(revision, safe="")
        + "/" + "/".join(filename_parts)
    )
    target = root / repo.replace("/", "__") / src_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return {"path": str(target.relative_to(root)), "sha256": digest,
                "bytes": target.stat().st_size, "download_status": "already_present",
                "source_repo": repo, "source_path": src_path, "revision": revision}
    headers = {"User-Agent": "ICWDS-reproducibility/1.0"}
    if token:
        headers["Authorization"] = "Bearer " + token
    tmp = target.with_suffix(target.suffix + ".partial")
    for attempt in range(3):
        sha = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
                with open(tmp, "wb") as dest:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        dest.write(block)
                        sha.update(block)
                        size += len(block)
            tmp.replace(target)
            return {"path": str(target.relative_to(root)), "sha256": sha.hexdigest(),
                    "bytes": size, "download_status": "downloaded",
                    "source_repo": repo, "source_path": src_path, "revision": revision}
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 2:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Download failed for {src_path}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List verified manifest paths; do not download")
    parser.add_argument("--group", choices=GROUPS + ("all",), default="reports")
    parser.add_argument("--output", default="downloads")
    parser.add_argument("--revision", default=None, help="Use immutable HF revision SHA instead of main")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifacts = [a for a in load_manifest()["artifacts"] if
                 args.group == "all" or a["group"] == args.group]
    if args.list or args.dry_run:
        for a in artifacts:
            print(f"[{a['group']}] [{a['status']}] {a['repo_id']}/{a['path']}")
        return 0
    token = os.environ.get("HF_TOKEN")
    output = pathlib.Path(args.output).resolve()
    locks = []
    revisions = {}
    for a in artifacts:
        key = (a["repo_id"], a["repo_type"])
        if key not in revisions:
            revisions[key] = args.revision or api_revision(*key, token) or "main"
        rev = revisions[key]
        print("Fetching", a["repo_id"], a["path"], "at", rev, flush=True)
        locks.append(download_file(a, output, rev, token))
    output.mkdir(parents=True, exist_ok=True)
    (output / "downloads.lock.json").write_text(json.dumps(
        {"schema_version": "1.0", "files": locks}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print("Saved", len(locks), "file entries to", output / "downloads.lock.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
