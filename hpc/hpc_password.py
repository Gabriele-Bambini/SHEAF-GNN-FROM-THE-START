"""Paramiko login helpers without hardcoded secrets.

Use this only when the OpenSSH key path is not available yet. Secrets are read
from the current process environment or from interactive prompts; they are never
written to repository files or command lines.
"""

from __future__ import annotations

import getpass
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import paramiko
from paramiko.ssh_exception import BadAuthenticationType, PartialAuthentication


DEFAULT_HOST = os.environ.get("TSGNN_HPC_HOSTNAME", "slogin.hpc.unibocconi.it")
DEFAULT_USER = os.environ.get("TSGNN_HPC_USER", "3393519")
DEFAULT_FALLBACK_USER = os.environ.get("TSGNN_HPC_FALLBACK_USER", "3393519@studbocconi.it")
DEFAULT_REMOTE_ROOT = os.environ.get("TSGNN_HPC_REMOTE_ROOT", "~/ts-gnn")
DEFAULT_KEY_PATH = Path(os.environ.get("TSGNN_HPC_KEY", "~/.ssh/id_bocconi")).expanduser()
DEFAULT_SECRET_PATH = Path(
    os.environ.get("TSGNN_HPC_SECRET_FILE", "~/.tsgnn/hpc_key_passphrase.dpapi")
).expanduser()


class HpcAuthenticationError(RuntimeError):
    """Raised when all configured HPC authentication attempts fail."""


def read_password() -> str:
    password = os.environ.get("TSGNN_HPC_PASSWORD")
    if password:
        return password
    cached = read_dpapi_secret(DEFAULT_SECRET_PATH)
    if cached:
        return cached
    return getpass.getpass("SSH key passphrase / HPC password: ")


def read_dpapi_secret(path: Path) -> str | None:
    if os.name != "nt" or not path.exists():
        return None

    script = f"""
    $ErrorActionPreference = 'Stop'
    $secure = Get-Content -LiteralPath {str(path)!r} | ConvertTo-SecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {{
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }} finally {{
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }}
    """
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.rstrip("\r\n")


def read_passkey() -> str:
    passkey = os.environ.get("TSGNN_HPC_PASSKEY")
    if passkey:
        return passkey
    return getpass.getpass("HPC passkey/MFA code: ")


def _default_users() -> tuple[str, ...]:
    users = [DEFAULT_USER, DEFAULT_FALLBACK_USER]
    return tuple(dict.fromkeys(user for user in users if user))


def _open_transport(host: str, port: int, timeout: float | None) -> paramiko.Transport:
    sock = socket.create_connection((host, port), timeout=timeout)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=timeout)
    return transport


def load_private_key(passphrase: str, key_path: Path = DEFAULT_KEY_PATH):
    if not key_path.exists():
        return None

    passphrase_bytes = passphrase.encode("utf-8") if isinstance(passphrase, str) else passphrase
    if hasattr(paramiko.PKey, "from_path"):
        try:
            return paramiko.PKey.from_path(str(key_path), passphrase=passphrase_bytes)
        except TypeError:
            pass
        except paramiko.PasswordRequiredException:
            raise
        except (paramiko.SSHException, OSError):
            pass

    loaders = (
        paramiko.Ed25519Key,
        paramiko.RSAKey,
        paramiko.ECDSAKey,
    )
    last_error: BaseException | None = None
    for loader in loaders:
        try:
            return loader.from_private_key_file(str(key_path), password=passphrase)
        except paramiko.PasswordRequiredException:
            raise
        except (paramiko.SSHException, OSError) as exc:
            last_error = exc
    if last_error is not None:
        print(
            f"Could not load private key at {key_path}: {type(last_error).__name__}. "
            "The passphrase did not decrypt this key; falling back to server password auth."
        )
    return None


def _prompt_needs_passkey(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(token in lowered for token in ("passkey", "otp", "mfa", "duo", "verification", "verify", "code", "token"))


def _prompt_needs_password(prompt: str) -> bool:
    return "password" in prompt.lower()


def _interactive_handler(password: str, passkey_reader: Callable[[], str]):
    def handler(title: str, instructions: str, prompts: list[tuple[str, bool]]) -> list[str]:
        responses: list[str] = []
        for prompt, _echo in prompts:
            if _prompt_needs_password(prompt):
                responses.append(password)
            elif _prompt_needs_passkey(prompt):
                responses.append(passkey_reader())
            else:
                responses.append(passkey_reader())
        return responses

    return handler


def _authenticate_transport(
    transport: paramiko.Transport,
    *,
    username: str,
    password: str,
    passkey_reader: Callable[[], str],
    pkey=None,
) -> str:
    methods_tried: list[str] = []
    allowed: Sequence[str] = ()

    try:
        allowed = transport.auth_none(username) or ()
        if transport.is_authenticated():
            return "none"
    except BadAuthenticationType as exc:
        allowed = exc.allowed_types
    except paramiko.AuthenticationException:
        allowed = ()

    if pkey is not None and (not allowed or "publickey" in allowed):
        try:
            methods_tried.append("publickey")
            transport.auth_publickey(username, pkey)
            if transport.is_authenticated():
                return "publickey"
        except paramiko.AuthenticationException:
            pass

    try:
        methods_tried.append("password")
        allowed = transport.auth_password(username, password, fallback=False) or ()
        if transport.is_authenticated():
            return "password"
    except PartialAuthentication as exc:
        allowed = exc.allowed_types
    except BadAuthenticationType as exc:
        allowed = exc.allowed_types
    except paramiko.AuthenticationException:
        allowed = allowed or ()

    if "keyboard-interactive" in allowed:
        methods_tried.append("keyboard-interactive")
        transport.auth_interactive(username, _interactive_handler(password, passkey_reader))
        if transport.is_authenticated():
            return "+".join(methods_tried)

    allowed_text = ",".join(allowed) if allowed else "unknown"
    tried_text = ",".join(methods_tried) if methods_tried else "none"
    raise paramiko.AuthenticationException(f"authentication rejected; allowed={allowed_text}; tried={tried_text}")


def connect(
    *,
    host: str = DEFAULT_HOST,
    port: int = 22,
    users: Sequence[str] | None = None,
    timeout: float | None = 30,
) -> paramiko.SSHClient:
    password = read_password()
    pkey = load_private_key(password)
    attempted: list[str] = []
    last_error: BaseException | None = None

    for username in users or _default_users():
        attempted.append(f"{username}@{host}")
        transport: paramiko.Transport | None = None
        try:
            transport = _open_transport(host, port, timeout)
            method = _authenticate_transport(
                transport,
                username=username,
                password=password,
                passkey_reader=read_passkey,
                pkey=pkey,
            )
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh._transport = transport
            print(f"Authenticated to {username}@{host} via {method}")
            return ssh
        except (paramiko.AuthenticationException, paramiko.SSHException, OSError) as exc:
            last_error = exc
            if transport is not None:
                transport.close()

    attempted_text = ", ".join(attempted)
    detail = f": {last_error}" if last_error else ""
    raise HpcAuthenticationError(f"HPC authentication failed for {attempted_text}{detail}") from last_error


def connect_password(*, host: str = DEFAULT_HOST, user: str = DEFAULT_USER) -> paramiko.SSHClient:
    """Backward-compatible wrapper for older scripts."""
    return connect(host=host, users=(user,))


def run(ssh: paramiko.SSHClient, command: str, *, check: bool = False) -> tuple[int, str, str]:
    print(f"$ ssh <hpc> {command}")
    _, out, err = ssh.exec_command(command)
    code = out.channel.recv_exit_status()
    stdout = out.read().decode("utf-8", errors="replace")
    stderr = err.read().decode("utf-8", errors="replace")
    if stdout:
        _safe_print(stdout)
    if stderr:
        _safe_print(stderr)
    if check and code != 0:
        raise RuntimeError(f"remote command failed with exit {code}: {command}")
    return code, stdout, stderr


def _safe_print(text: str) -> None:
    try:
        print(text, end="")
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))


def upload_files(ssh: paramiko.SSHClient, files: Iterable[Path], remote_root: str, *, base: Path | None = None) -> int:
    sftp = ssh.open_sftp()
    count = 0
    try:
        for path in files:
            path = Path(path)
            rel = path.relative_to(base).as_posix() if base is not None else path.name
            remote_path = f"{remote_root.rstrip('/')}/{rel}"
            remote_dir = str(Path(remote_path).parent).replace("\\", "/")
            run(ssh, f"mkdir -p {remote_dir}", check=True)
            sftp.put(str(path), remote_path)
            count += 1
    finally:
        sftp.close()
    return count
