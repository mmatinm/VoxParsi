"""Voice cloning is disabled in this project."""

from __future__ import annotations

import sys


def main() -> int:
    print("Voice cloning is disabled in this project.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
