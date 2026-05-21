from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    helper = Path(__file__).with_name("sync_and_submit.py")
    raise SystemExit(subprocess.call([sys.executable, str(helper), "probe"]))
