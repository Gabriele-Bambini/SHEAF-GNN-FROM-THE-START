"""Download HPC results JSON files via paramiko."""
import paramiko, os

HOST = "slogin.hpc.unibocconi.it"
USER = "3393519"
PASS = "GrabdeUomo1!"
REMOTE = "/home/3393519/ts-gnn/results"
LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hpc_496333")

os.makedirs(LOCAL, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)
sftp = ssh.open_sftp()

TARGETS = [
    "benchmarks.json", "allele_swap.json", "metabric_enrichment.csv",
    "nature_run_summary.json", "genes_intersect.json", "ablation_table.tex",
    "holdout.json", "multiseed_aggregate.json",
]

def scan_and_download(remote_dir, prefix=""):
    """Recursively scan remote dir and download target files."""
    n = 0
    try:
        items = sftp.listdir_attr(remote_dir)
    except Exception as e:
        print(f"  Cannot list {remote_dir}: {e}")
        return n
    for attr in items:
        rpath = f"{remote_dir}/{attr.filename}"
        if attr.filename in TARGETS:
            local_name = f"{prefix}{attr.filename}" if prefix else attr.filename
            local_path = os.path.join(LOCAL, local_name)
            sftp.get(rpath, local_path)
            print(f"  OK: {local_name} ({attr.st_size} bytes)")
            n += 1
        # Recurse into subdirs
        try:
            sftp.listdir(rpath)
            sub_prefix = f"{prefix}{attr.filename}__"
            n += scan_and_download(rpath, sub_prefix)
        except:
            pass
    return n

print(f"Scanning {REMOTE} ...")
total = scan_and_download(REMOTE)
print(f"\nDownloaded {total} files to {LOCAL}")

sftp.close()
ssh.close()
