"""Publish src/features.py to Kaggle as an importable dataset.

Kaggle kernels cannot import from this repo, so the shared feature code has to
live on Kaggle. Two ways were tried:

  * as a utility *kernel* via `kernel_sources` -- does NOT work for script
    kernels. Kaggle mounts it at /kaggle/input/<slug>/__script__.py, renamed
    and not on sys.path, so `import rsna_features` fails.
  * as a *dataset* -- works. The filename is preserved, so the consumer adds
    its directory to sys.path and imports normally.

A copy is unavoidable, so it is made generated rather than authored:
src/features.py is the only real version, the copy carries a DO-NOT-EDIT header
with the source hash, and this script re-syncs before every push.

    python3 tools/sync_utility.py         # sync only
    python3 tools/sync_utility.py --push  # sync then publish a new version
"""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST_DIR = ROOT / "kaggle" / "utility_features"
# (source in src/, name on Kaggle) — both ship in one dataset
MODULES = [("features.py", "rsna_features.py"),
           ("topology.py", "rsna_topology.py"),
           ("persistence.py", "rsna_persistence.py"),
           ("eikonal.py", "rsna_eikonal.py")]
SLUG = "catejsj/rsna-features-code"

HEADER = """# GENERATED FILE - DO NOT EDIT.
# Copied from src/ by tools/sync_utility.py.
# Edit the source, then re-run the sync. Source sha256: {sha}
"""


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-400:])
    return r.returncode


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in MODULES:
        body = (ROOT / "src" / src_name).read_text(encoding="utf-8")
        sha = hashlib.sha256(body.encode()).hexdigest()[:16]
        (DEST_DIR / dest_name).write_text(HEADER.format(sha=sha) + body,
                                          encoding="utf-8")
        print(f"synced src/{src_name} -> {dest_name}  (sha {sha})")

    meta = DEST_DIR / "dataset-metadata.json"
    meta.write_text(json.dumps(
        {"title": "rsna-features-code", "id": SLUG,
         "licenses": [{"name": "other"}]}, indent=2) + "\n")

    # the kernel-metadata from the earlier attempt would confuse a dataset push
    stale = DEST_DIR / "kernel-metadata.json"
    if stale.exists():
        stale.unlink()

    if "--push" not in sys.argv:
        return 0

    code = run(["kaggle", "datasets", "version", "-p", str(DEST_DIR),
                "-m", "sync src/features.py + src/topology.py"])
    if code != 0:
        print("version failed (dataset may not exist yet) -- trying create")
        code = run(["kaggle", "datasets", "create", "-p", str(DEST_DIR)])
    return code


if __name__ == "__main__":
    sys.exit(main())
