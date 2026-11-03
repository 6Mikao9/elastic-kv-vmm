"""Verify every recorded Phase 2 source hash against current or archived files."""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    # Git inventory respects ignored caches/logs and never traverses outside the repo.
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    digests = {}
    manifests = []
    for name in set(paths) - {""}:
        path = ROOT / name
        if path.suffix in (".py", ".cpp", ".sh", ".md"):
            digests.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(name)
        if name.startswith(("results/phase2/", "results/vllm/")) and path.name == "source_manifest.json":
            manifests.append(path)
    missing = []
    count = 0
    for path in sorted(manifests):
        for source, digest in json.loads(path.read_text(encoding="utf-8"))["sha256"].items():
            count += 1
            if digest not in digests:
                missing.append({"manifest": path.relative_to(ROOT).as_posix(),
                                "source": source, "sha256": digest})
    print(json.dumps({"manifests": len(manifests), "source_references": count,
                      "missing": missing}, indent=2))
    return bool(missing)


if __name__ == "__main__":
    raise SystemExit(main())
