"""Administrator bootstrap. Routine CD can only transfer and activate agent images."""

import argparse
import os
import pwd
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/opt/firebridge/agent")
ACCOUNT = "firebridge-agent-deploy"
HELPER = "/usr/local/sbin/firebridge-agent-gateway"


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def provision(public_key, runtime_env):
    if os.geteuid() != 0:
        raise RuntimeError("Run in an administrator root session on the deployment server.")
    key = public_key.read_text().strip()
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]+)?", key):
        raise ValueError("Expected one Ed25519 public key.")
    run("ssh-keygen", "-lf", str(public_key))
    run("docker", "network", "inspect", "firebridge-webapp_web")
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    ROOT.chmod(0o700)
    runtime = ROOT / "runtime.env"
    if not runtime.exists():
        if runtime_env is None:
            raise ValueError("First installation requires --runtime-env with private production credentials.")
        shutil.copyfile(runtime_env, runtime)
    runtime.chmod(0o600)
    directory = ROOT / "data"
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chown(directory, 10001, 10001)
    directory.chmod(0o700)
    source = Path(__file__).resolve().parent
    shutil.copyfile(source / "compose.production.yaml", ROOT / "compose.yaml")
    shutil.copyfile(source / "gateway.py", HELPER)
    os.chmod(HELPER, 0o755)
    os.chown(HELPER, 0, 0)
    try:
        entry = pwd.getpwnam(ACCOUNT)
    except KeyError:
        run(
            "useradd", "--system", "--create-home", "--home-dir", "/var/lib/" + ACCOUNT, "--shell", "/bin/bash", ACCOUNT
        )
        entry = pwd.getpwnam(ACCOUNT)
    ssh_directory = Path(entry.pw_dir) / ".ssh"
    ssh_directory.mkdir(mode=0o700, exist_ok=True)
    ssh_directory.chmod(0o700)
    os.chown(ssh_directory, entry.pw_uid, entry.pw_gid)
    authorized = ssh_directory / "authorized_keys"
    authorized.write_text(f'restrict,command="/usr/bin/sudo -n {HELPER}" {key}\n')
    authorized.chmod(0o600)
    os.chown(authorized, entry.pw_uid, entry.pw_gid)
    rule = Path("/etc/sudoers.d/firebridge-agent-cd")
    temporary = rule.with_suffix(".partial")
    temporary.write_text(
        f'Defaults:{ACCOUNT} env_keep += "SSH_ORIGINAL_COMMAND"\n{ACCOUNT} ALL=(root) NOPASSWD: {HELPER} ""\n'
    )
    temporary.chmod(0o440)
    run("visudo", "-cf", str(temporary))
    temporary.replace(rule)
    config = Path("/etc/ssh/sshd_config.d/00-firebridge.conf")
    original = config.read_text()
    match = re.search(r"^AllowUsers (.+)$", original, re.M)
    if not match:
        raise ValueError("The server SSH AllowUsers policy needs administrator review.")
    allowed = match[1].split()
    if ACCOUNT not in allowed:
        allowed.append(ACCOUNT)
        config.write_text(original[: match.start()] + "AllowUsers " + " ".join(allowed) + original[match.end() :])
        try:
            run("/usr/sbin/sshd", "-t")
        except Exception:
            config.write_text(original)
            raise
        run("systemctl", "reload", "ssh")
    print("Agent runtime and restricted deployment account installed; existing data services remain running.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path)
    args = parser.parse_args()
    provision(args.public_key, args.runtime_env)
