"""Verify the installed Streamlit wheel's JavaScript against TN-2026111186000044.

Offline inspection, not a replacement for the CSIRT's external rescan.
Exit nonzero for vulnerable, missing, ambiguous or unrecognized bundles.
"""

from __future__ import annotations

import argparse
import re
from importlib.metadata import distribution
from pathlib import Path

MINIMUM = {"DOMPurify": (3, 4, 17), "moment.js": (2, 31, 0)}


def inspect_assets(root: Path):
    findings = {name: [] for name in MINIMUM}
    unknown = []
    for path in root.rglob("*.js"):
        text = path.read_text(encoding="utf-8")
        patterns = {
            "DOMPurify": (
                "purify" in path.name.lower() or "DOMPurify" in text,
                r'\.version\s*=\s*["\x27`]([0-9]+\.[0-9]+\.[0-9]+)["\x27`]\s*,\s*\w+\.removed',
            ),
            "moment.js": (
                "moment" in path.name.lower() or "moment.js" in text,
                r'\.version\s*=\s*["\x27`]([0-9]+\.[0-9]+\.[0-9]+)["\x27`]\s*,[^;]{0,80}\.fn\s*=',
            ),
        }
        for library, (relevant, pattern) in patterns.items():
            versions = re.findall(pattern, text)
            # Inspect assignments regardless of filenames (bundles can be renamed).
            if versions:
                findings[library].extend((str(path.relative_to(root)), v) for v in versions)
            elif relevant:
                # Imports/identifiers can mention a library without embedding it.
                if library == "DOMPurify" and ("sanitize=" in text or "sanitize =" in text):
                    unknown.append((library, str(path.relative_to(root))))
                if library == "moment.js" and ("moment.js" in text and "version" in text):
                    unknown.append((library, str(path.relative_to(root))))
    return findings, unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", type=Path)
    args = parser.parse_args()
    dist = distribution("streamlit") if args.static_dir is None else None
    root = args.static_dir or Path(dist.locate_file("streamlit/static"))
    if not root.is_dir():
        raise SystemExit("FAIL: Streamlit static directory was not found")
    if dist:
        print(f"Streamlit {dist.version}")
    findings, unknown = inspect_assets(root)
    ok = not unknown
    for library, minimum in MINIMUM.items():
        rows = findings[library]
        if not rows:
            print(f"FAIL: {library} version was not verifiable")
            ok = False
        for path, version in rows:
            valid = tuple(map(int, version.split("."))) >= minimum
            ok &= valid
            print(f"{'PASS' if valid else 'FAIL'}: {library} {version} — {path}")
    for library, path in unknown:
        print(f"FAIL: unrecognized {library} implementation — {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
