"""Verify immutable source, data, split, and checkpoint hashes."""

import hashlib
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "release_manifest.json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    failures = []
    for relative_path, expected in manifest["sha256"].items():
        path = os.path.join(PROJECT_ROOT, *relative_path.split("/"))
        if not os.path.isfile(path):
            failures.append("missing: {}".format(relative_path))
            continue
        observed = sha256(path)
        if observed.lower() != expected.lower():
            failures.append(
                "hash mismatch: {} expected={} observed={}".format(
                    relative_path, expected, observed
                )
            )
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        raise SystemExit(1)
    print("Release verification passed for {} files.".format(len(manifest["sha256"])))


if __name__ == "__main__":
    main()

