"""List the DVC cache objects used by the current workspace, one path per line,
relative to the cache's ``files/`` directory (e.g. ``md5/ab/cdef...``).

Intended as input for ``rclone copy --files-from`` as an alternative to
``dvc push``; the DVC 3 remote layout mirrors the local cache layout.
Unlike ``dvc push``, outputs whose cache cannot be collected (e.g. a missing
``.dir`` object) are skipped with a warning instead of aborting.

Only objects present in the local cache are listed.

Usage: uv run python src/dvc_used_objects.py [targets...] > objects.txt
"""

import os
import sys

from dvc.exceptions import DvcException
from dvc.repo import Repo


def _rel(oid):
    return f"md5/{oid[:2]}/{oid[2:]}"


def main(targets):
    repo = Repo()
    cache_root = repo.cache.local.path  # .../files/md5
    paths, skipped, missing = set(), [], 0
    for stage, filter_info in repo.index.collect_targets(targets or None):
        for out in stage.outs:
            if not out.use_cache:
                continue
            try:
                used = out.get_used_objs(filter_info=filter_info)
            except DvcException as exc:
                skipped.append(f"{out} ({exc})")
                continue
            for objs in used.values():
                for hash_info in objs:
                    rel = _rel(hash_info.value)
                    if os.path.exists(os.path.join(cache_root, rel[4:])):
                        paths.add(rel)
                    else:
                        missing += 1
    for path in sorted(paths):
        print(path)
    for s in skipped:
        print(f"skipped: {s}", file=sys.stderr)
    print(
        f"{len(paths)} objects, {missing} missing from local cache, "
        f"{len(skipped)} outputs skipped",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
