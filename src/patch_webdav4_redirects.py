"""Patch the installed webdav4 package so PROPFIND requests follow redirects.

The Hetzner storage box WebDAV server (Apache) answers requests for a
directory without a trailing slash with "301 Moved Permanently".
webdav4 (used by dvc-webdav) does not follow redirects in ``Client.propfind``,
so DVC fails with ``received 301 (Moved Permanently)`` when checking
whether remote directories exist. This script flips the default of
``follow_redirects`` in ``Client.propfind`` to ``True``.

The patch lives inside the virtual environment, so re-run this script after
recreating the venv or reinstalling webdav4:

    uv run python src/patch_webdav4_redirects.py

Use ``--check`` to only report whether the patch is applied (exit code 1 if not).
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

MARKER = "# patched: follow redirects (Hetzner WebDAV)"

PROPFIND_PATTERN = re.compile(
    r"(def propfind\(.*?follow_redirects: bool = )(False|True),[^\n]*",
    re.DOTALL,
)


def find_client_file() -> Path:
    spec = importlib.util.find_spec("webdav4.client")
    if spec is None or spec.origin is None:
        sys.exit("webdav4 is not installed in this environment (install dvc-webdav).")
    return Path(spec.origin)


def is_patched(source: str) -> bool:
    match = PROPFIND_PATTERN.search(source)
    return bool(match) and match.group(2) == "True"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="only check whether the patch is applied"
    )
    args = parser.parse_args()

    client_file = find_client_file()
    source = client_file.read_text()

    if PROPFIND_PATTERN.search(source) is None:
        print(
            f"Could not find Client.propfind's follow_redirects default in {client_file}; "
            "webdav4 may have changed. Inspect the file manually.",
            file=sys.stderr,
        )
        return 2

    if is_patched(source):
        print(f"Already patched: {client_file}")
        return 0

    if args.check:
        print(f"Not patched: {client_file}")
        return 1

    patched = PROPFIND_PATTERN.sub(rf"\g<1>True,  {MARKER}", source, count=1)
    client_file.write_text(patched)

    # Remove stale bytecode so the change takes effect immediately
    for pyc in (client_file.parent / "__pycache__").glob("client.*.pyc"):
        pyc.unlink()

    print(f"Patched: {client_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
