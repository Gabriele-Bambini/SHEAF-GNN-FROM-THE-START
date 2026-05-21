from __future__ import annotations

from hpc_ssh import run_print


if __name__ == "__main__":
    run_print("sinfo -o '%P %a %l %D %t %G' 2>&1 | head -40", check=False)
