from __future__ import annotations

import sys
from pathlib import Path

import paramiko
import pytest


ROOT = Path(__file__).resolve().parents[1]
HPC_DIR = ROOT / "hpc"
if str(HPC_DIR) not in sys.path:
    sys.path.insert(0, str(HPC_DIR))

import hpc_password  # noqa: E402


class FakeTransport:
    def __init__(
        self,
        outcomes: list[str],
        *,
        interactive_prompts: list[tuple[str, bool]] | None = None,
    ) -> None:
        self.outcomes = list(outcomes)
        self.interactive_prompts = interactive_prompts or []
        self.started_client = False
        self.auth_calls: list[tuple[str, str, object]] = []
        self.active = True
        self.authenticated = False
        self.allowed_types = ["password", "keyboard-interactive"]

    def start_client(self, timeout=None):  # noqa: ANN001
        self.started_client = True

    def is_authenticated(self) -> bool:
        return self.authenticated

    def auth_none(self, username: str):
        raise paramiko.BadAuthenticationType("none rejected", self.allowed_types)

    def auth_password(self, username: str, password: str, fallback: bool = False):
        self.auth_calls.append(("password", username, password))
        outcome = self.outcomes.pop(0)
        if outcome == "authenticated":
            self.authenticated = True
            return []
        if outcome == "partial":
            return ["keyboard-interactive"]
        raise paramiko.AuthenticationException("password failed")

    def auth_publickey(self, username: str, pkey):
        self.auth_calls.append(("publickey", username, pkey))
        outcome = self.outcomes.pop(0)
        if outcome == "authenticated":
            self.authenticated = True
            return []
        raise paramiko.AuthenticationException("publickey failed")

    def auth_interactive(self, username: str, handler, submethods: str = ""):
        responses = handler("MFA", "Enter code", self.interactive_prompts)
        self.auth_calls.append(("interactive", username, tuple(responses)))
        outcome = self.outcomes.pop(0)
        if outcome == "authenticated":
            self.authenticated = True
            return []
        raise paramiko.AuthenticationException("interactive failed")

    def close(self) -> None:
        self.active = False


def test_connect_uses_password_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    transports = [FakeTransport(["authenticated"])]
    monkeypatch.setattr(hpc_password, "read_password", lambda: "pw")
    monkeypatch.setattr(hpc_password, "read_passkey", lambda: "otp")
    monkeypatch.setattr(hpc_password, "load_private_key", lambda passphrase: None)
    monkeypatch.setattr(hpc_password, "_open_transport", lambda *args, **kwargs: transports.pop(0))

    client = hpc_password.connect(host="host", users=("mail@example.test",))

    assert isinstance(client, paramiko.SSHClient)
    assert client.get_transport().auth_calls == [("password", "mail@example.test", "pw")]


def test_read_password_prefers_dpapi_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TSGNN_HPC_PASSWORD", raising=False)
    monkeypatch.setattr(hpc_password, "DEFAULT_SECRET_PATH", Path("secret.dpapi"))
    monkeypatch.setattr(hpc_password, "read_dpapi_secret", lambda path: "cached-passphrase")

    assert hpc_password.read_password() == "cached-passphrase"


def test_connect_answers_keyboard_interactive_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport(
        ["partial", "authenticated"],
        interactive_prompts=[("Verification code:", False), ("Password:", False)],
    )
    monkeypatch.setattr(hpc_password, "read_password", lambda: "pw")
    monkeypatch.setattr(hpc_password, "read_passkey", lambda: "otp")
    monkeypatch.setattr(hpc_password, "load_private_key", lambda passphrase: None)
    monkeypatch.setattr(hpc_password, "_open_transport", lambda *args, **kwargs: transport)

    hpc_password.connect(host="host", users=("mail@example.test",))

    assert transport.auth_calls == [
        ("password", "mail@example.test", "pw"),
        ("interactive", "mail@example.test", ("otp", "pw")),
    ]


def test_connect_tries_private_key_before_password(monkeypatch: pytest.MonkeyPatch) -> None:
    key = object()
    transport = FakeTransport(["authenticated"])
    transport.allowed_types = ["publickey", "password"]
    monkeypatch.setattr(hpc_password, "read_password", lambda: "key-passphrase")
    monkeypatch.setattr(hpc_password, "read_passkey", lambda: "otp")
    monkeypatch.setattr(hpc_password, "load_private_key", lambda passphrase: key)
    monkeypatch.setattr(hpc_password, "_open_transport", lambda *args, **kwargs: transport)

    hpc_password.connect(host="host", users=("3393519",))

    assert transport.auth_calls == [("publickey", "3393519", key)]


def test_connect_falls_back_to_second_user(monkeypatch: pytest.MonkeyPatch) -> None:
    first = FakeTransport(["fail", "fail"])
    second = FakeTransport(["authenticated"])
    transports = [first, second]
    monkeypatch.setattr(hpc_password, "read_password", lambda: "pw")
    monkeypatch.setattr(hpc_password, "read_passkey", lambda: "otp")
    monkeypatch.setattr(hpc_password, "load_private_key", lambda passphrase: None)
    monkeypatch.setattr(hpc_password, "_open_transport", lambda *args, **kwargs: transports.pop(0))

    hpc_password.connect(host="host", users=("mail@example.test", "3393519"))

    assert first.active is False
    assert second.auth_calls == [("password", "3393519", "pw")]


def test_auth_failure_message_does_not_include_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hpc_password, "read_password", lambda: "super-secret-password")
    monkeypatch.setattr(hpc_password, "read_passkey", lambda: "super-secret-otp")
    monkeypatch.setattr(hpc_password, "load_private_key", lambda passphrase: None)
    monkeypatch.setattr(hpc_password, "_open_transport", lambda *args, **kwargs: FakeTransport(["fail", "fail"]))

    with pytest.raises(hpc_password.HpcAuthenticationError) as excinfo:
        hpc_password.connect(host="host", users=("mail@example.test",))

    message = str(excinfo.value)
    assert "mail@example.test@host" in message
    assert "super-secret-password" not in message
    assert "super-secret-otp" not in message


def test_password_only_server_does_not_try_keyboard_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport(["fail"])
    transport.allowed_types = ["password"]
    monkeypatch.setattr(hpc_password, "read_password", lambda: "pw")
    monkeypatch.setattr(hpc_password, "read_passkey", lambda: "otp")
    monkeypatch.setattr(hpc_password, "load_private_key", lambda passphrase: None)
    monkeypatch.setattr(hpc_password, "_open_transport", lambda *args, **kwargs: transport)

    with pytest.raises(hpc_password.HpcAuthenticationError) as excinfo:
        hpc_password.connect(host="host", users=("mail@example.test",))

    assert transport.auth_calls == [("password", "mail@example.test", "pw")]
    assert "allowed=password" in str(excinfo.value)
