from __future__ import annotations

from hpc_ssh import run_print


if __name__ == "__main__":
    run_print("whoami && hostname && sinfo -o '%P %a %l %D %t' 2>&1 | head -30 && sacctmgr show user $(whoami) withassoc 2>&1 | head -40", check=False)
