#!/usr/bin/env python3
"""strata-m4 - the command for the Mistral Small 4 engine, in the manner of Strata's own `strata` command.

    strata-m4 launch            start the server in the background (if it is not running) and open the monitor
    strata-m4 serve             run the server in the foreground (Ctrl+C stops it)
    strata-m4 stop              stop the server (graceful, then forced after --grace seconds)
    strata-m4 restart           stop, then launch without the monitor
    strata-m4 status [--json]   one look: state, model, context, queue, rates, GPU, placement
    strata-m4 monitor           Strata's monitor on this server (key and port from m4.env)
    strata-m4 logs [-f] [-n N]  the server log
    strata-m4 config [show | edit | set NAME VALUE | args add|remove OPTION...]
    strata-m4 doctor            read-only checks: GPU, files, venv, monitor, port
    strata-m4 bench | speed | sanity | parity | longctx [options]   the measuring scripts (server stopped first)

Settings live in ~/strata-m4/m4.env (M4_MODEL, M4_TOKENIZER, M4_ENGINE_ARGS, M4_PORT, M4_HOST, M4_API_KEY, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("M4_HOME", Path(__file__).resolve().parent))
ENV = HOME / "m4.env"
LOGS = HOME / "logs"
RUN = HOME / "run"
PIDFILE = RUN / "m4serve.pid"
SCRIPTS = ("bench", "speed", "sanity", "parity", "longctx")


# ---------------------------------------------------------------------------------------------- settings
def read_env() -> dict[str, str]:
    """m4.env as bash would see it (sourced in a clean shell, so quoting and $VARS behave the same)."""
    if not ENV.exists():
        sys.exit(f"strata-m4: {ENV} is missing - run install.sh first")
    out = subprocess.run(["bash", "-c", f"set -a; source {shlex.quote(str(ENV))}; env -0"], capture_output=True,
                         check=True).stdout
    env = {}
    for item in out.split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            if k.startswith(b"M4_") or k == b"STRATA_API_KEY":
                env[k.decode()] = v.decode(errors="replace")
    return env


def port(env) -> int:
    return int(env.get("M4_PORT") or 8095)


def key(env) -> str | None:
    k = env.get("M4_API_KEY") or os.environ.get("M4_API_KEY") or env.get("STRATA_API_KEY") or os.environ.get("STRATA_API_KEY")
    return None if not k or k.strip().lower() == "none" else k.strip()


def get(env, path: str, timeout: float = 3.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port(env)}{path}")
    k = key(env)
    if k:
        req.add_header("Authorization", "Bearer " + k)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def healthy(env) -> dict | None:
    try:
        return get(env, "/health", 2.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------- processes
def server_pids() -> list[int]:
    pids = []
    if PIDFILE.exists():
        try:
            p = int(PIDFILE.read_text().strip())
            os.kill(p, 0)
            pids.append(p)
        except (ValueError, ProcessLookupError, PermissionError):
            PIDFILE.unlink(missing_ok=True)
    for p in procs(lambda argv: Path(argv[0]).name.startswith("python") and str(HOME / "m4serve.py") in argv[1:3]):
        if p not in pids:
            pids.append(p)
    return pids


def procs(match) -> list[int]:
    """Processes of this user whose argv matches (read from /proc: no pattern can match a shell that merely
    mentions the path)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors="replace") for x in (d / "cmdline").read_bytes().split(b"\0") if x]
        except OSError:
            continue
        if argv and match(argv):
            out.append(int(d.name))
    return out


def engine_pids() -> list[int]:
    return procs(lambda argv: argv[0] == str(HOME / "bin" / "strata-m4"))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def start_background(env) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    log = LOGS / time.strftime("server-%Y%m%d-%H%M%S.log")
    link = LOGS / "server.log"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(log.name)
    except OSError:
        pass
    f = open(log, "ab", buffering=0)
    p = subprocess.Popen([str(HOME / "start-m4.sh")], stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True)
    PIDFILE.write_text(str(p.pid))
    return p.pid


def wait_ready(env, pid: int, limit: float = 600.0) -> bool:
    """Polls /health while showing the load's milestones from the log; False when the server died or timed out."""
    log = LOGS / "server.log"
    seen = 0
    t0 = time.time()
    keys = ("experts:", "EAGLE head", "pinned", "latent KV", "default system prompt", "listening", "error", "Traceback",
            "failed", "fail:")
    while time.time() - t0 < limit:
        try:
            lines = log.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines[seen:]:
            if any(k in line for k in keys):
                print("  " + line.strip()[:160], flush=True)
        seen = len(lines)
        if healthy(env):
            return True
        if not alive(pid):
            print("strata-m4: the server stopped during the start; the last lines of", log)
            for line in lines[-15:]:
                print("  " + line)
            return False
        time.sleep(1.0)
    print(f"strata-m4: not ready after {limit:.0f} s; see {log}")
    return False


def stop(grace: float = 30.0) -> bool:
    pids = server_pids()
    if not pids and not engine_pids():
        print("strata-m4: no server is running")
        return True
    for p in pids:
        try:
            os.kill(p, signal.SIGINT)
        except ProcessLookupError:
            pass
    t0 = time.time()
    while time.time() - t0 < grace and (any(alive(p) for p in pids) or engine_pids()):
        time.sleep(0.5)
    for p in pids:
        if alive(p):
            print(f"strata-m4: m4serve {p} did not stop in {grace:.0f} s; SIGTERM, then SIGKILL")
            try:
                os.kill(p, signal.SIGTERM)
                time.sleep(5)
                if alive(p):
                    os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(1)
    for p in engine_pids():   # an engine whose server is gone
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    PIDFILE.unlink(missing_ok=True)
    print("strata-m4: stopped")
    return True


# ---------------------------------------------------------------------------------------------- commands
def cmd_launch(a, env) -> int:
    if healthy(env):
        print(f"strata-m4: the server is running on port {port(env)}")
    else:
        if server_pids():
            print("strata-m4: a server process exists but does not answer; stopping it first")
            stop(a.grace)
        print(f"strata-m4: starting the server on port {port(env)} (log: {LOGS / 'server.log'})")
        pid = start_background(env)
        if not wait_ready(env, pid, a.timeout):
            return 1
        print(f"strata-m4: ready - http://{env.get('M4_HOST', '127.0.0.1')}:{port(env)}/v1")
    if a.no_monitor:
        return 0
    return cmd_monitor(argparse.Namespace(rest=[]), env)


def cmd_serve(a, env) -> int:
    if healthy(env) or server_pids():
        sys.exit("strata-m4: a server is already running (strata-m4 stop)")
    os.execv(str(HOME / "start-m4.sh"), [str(HOME / "start-m4.sh"), *a.rest])
    return 0


def cmd_stop(a, env) -> int:
    return 0 if stop(a.grace) else 1


def cmd_restart(a, env) -> int:
    stop(a.grace)
    a.no_monitor = True
    return cmd_launch(a, env)


def fmt_gib(b) -> str:
    return f"{b / 2**30:.1f}" if isinstance(b, (int, float)) else "--"


def cmd_status(a, env) -> int:
    h = healthy(env)
    if not h:
        state = "starting (process up, not answering yet)" if server_pids() else "stopped"
        if a.json:
            print(json.dumps({"state": state, "port": port(env)}))
        else:
            print(f"strata-m4: {state} (port {port(env)})")
        return 3
    try:
        t = get(env, "/v1/telemetry")
    except urllib.error.HTTPError as e:
        sys.exit(f"strata-m4: telemetry refused ({e.code}); is M4_API_KEY in m4.env the server's key?")
    if a.json:
        print(json.dumps({"health": h, "telemetry": t}, indent=1))
        return 0
    eng, w, r = h.get("engine", {}), t.get("work", {}), t.get("rates", {})
    dev = (t.get("devices") or [{}])[0]
    up = time.time() - (t.get("instance", {}).get("boot_time") or time.time())
    print(f"strata-m4: ready on port {port(env)}, up {up / 3600:.1f} h, pid {','.join(map(str, server_pids())) or '?'}")
    print(f"  model      {t.get('model', {}).get('id')}  context {eng.get('ctx')}  kv {eng.get('kv')}  "
          f"eagle {'on' if eng.get('eagle') else 'off'}")
    print(f"  placement  {eng.get('placement', '--')}")
    slot = (w.get("slots") or [{}])[0]
    print(f"  work       {w.get('active')} running ({slot.get('state')}), {w.get('queued')} waiting "
          f"(limit {w.get('max_queue')}), rejected {w.get('admission_rejected')}, timeouts {w.get('queue_timeouts')}")
    print(f"  rates      answer {r.get('answer_tok_s')} tok/s, prompt {r.get('prefill_tok_s')} tok/s, "
          f"{t.get('totals', {}).get('requests')} requests, {t.get('totals', {}).get('output_tokens')} tokens out")
    print(f"  gpu        {dev.get('name')}  {fmt_gib(dev.get('mem_used'))}/{fmt_gib(dev.get('mem_total'))} GiB  "
          f"{dev.get('util')} %  {dev.get('temp')} C  {dev.get('power')} W")
    ram = t.get("ram", {})
    print(f"  ram        {fmt_gib(ram.get('used'))}/{fmt_gib(ram.get('total'))} GiB")
    return 0


def cmd_monitor(a, env) -> int:
    if not (HOME / "monitor-m4.sh").exists():
        sys.exit(f"strata-m4: {HOME / 'monitor-m4.sh'} is missing (install.sh copies it)")
    os.execv(str(HOME / "monitor-m4.sh"), [str(HOME / "monitor-m4.sh"), *getattr(a, "rest", [])])
    return 0


def cmd_logs(a, env) -> int:
    log = LOGS / "server.log"
    if not log.exists():
        sys.exit(f"strata-m4: no server log yet ({log})")
    args = ["tail", "-n", str(a.n)] + (["-F"] if a.follow else []) + [str(log)]
    os.execvp("tail", args)
    return 0


def set_line(text: str, name: str, value: str) -> str:
    line = f'{name}="{value}"' if (" " in value or not value) else f"{name}={value}"
    pat = re.compile(rf"^#?\s*{re.escape(name)}=.*$", re.M)
    return pat.sub(line, text, count=1) if pat.search(text) else text.rstrip("\n") + "\n" + line + "\n"


def cmd_config(a, env) -> int:
    text = ENV.read_text()
    if a.action in (None, "show"):
        for line in text.splitlines():
            if re.match(r"^\s*(M4_API_KEY|STRATA_API_KEY)=", line):
                line = line.split("=", 1)[0] + "=***"
            print(line)
        return 0
    if a.action == "edit":
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
        return subprocess.call([editor, str(ENV)])
    if a.action == "set":
        if len(a.values) != 2:
            sys.exit("strata-m4 config set NAME VALUE   (e.g. M4_PORT 8095)")
        name, value = a.values
        if not re.fullmatch(r"(M4_[A-Z_]+|STRATA_API_KEY)", name):
            sys.exit("strata-m4: only M4_* names (and STRATA_API_KEY) belong in m4.env")
        shutil.copy2(ENV, ENV.with_suffix(".env.bak"))
        ENV.write_text(set_line(text, name, value))
        os.chmod(ENV, 0o600)
        print(f"strata-m4: {name} set (old file: {ENV.with_suffix('.env.bak')}); takes effect with strata-m4 restart")
        return 0
    if a.action == "args":
        if len(a.values) < 2 or a.values[0] not in ("add", "remove"):
            sys.exit("strata-m4 config args add|remove OPTION [VALUE]   (e.g. args add --chunk 4096)")
        cur = shlex.split(env.get("M4_ENGINE_ARGS", ""))
        opt, val = a.values[1], a.values[2:]
        out, i = [], 0
        while i < len(cur):   # drop the option and its value (an argument that does not start with --)
            if cur[i] == opt:
                i += 1
                while i < len(cur) and not cur[i].startswith("--"):
                    i += 1
                continue
            out.append(cur[i])
            i += 1
        if a.values[0] == "add":
            out += [opt, *val]
        shutil.copy2(ENV, ENV.with_suffix(".env.bak"))
        ENV.write_text(set_line(text, "M4_ENGINE_ARGS", " ".join(shlex.quote(x) for x in out)))
        os.chmod(ENV, 0o600)
        print("strata-m4: M4_ENGINE_ARGS =", " ".join(out))
        print("takes effect with strata-m4 restart")
        return 0
    sys.exit("strata-m4 config [show | edit | set NAME VALUE | args add|remove OPTION [VALUE]]")


def cmd_doctor(a, env) -> int:
    ok = True

    def line(good: bool, what: str, detail: str = ""):
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {what}" + (f": {detail}" if detail else ""))
    print("strata-m4 doctor")
    line((HOME / "bin" / "strata-m4").exists(), "engine binary", str(HOME / "bin" / "strata-m4"))
    line((HOME / "venv" / "bin" / "python").exists(), "python venv", str(HOME / "venv"))
    m = env.get("M4_MODEL", "")
    line(bool(m) and Path(m).exists(), "model (M4_MODEL)", m or "not set")
    ea = shlex.split(env.get("M4_ENGINE_ARGS", ""))
    if "--eagle" in ea:
        p = ea[ea.index("--eagle") + 1] if ea.index("--eagle") + 1 < len(ea) else ""
        line(Path(p).exists(), "EAGLE head", p)
    st = oct(ENV.stat().st_mode & 0o777)
    line(not key(env) or (ENV.stat().st_mode & 0o077) == 0, "m4.env permissions", st)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        line(bool(q), "GPU", q)
    except Exception as e:
        line(False, "GPU (nvidia-smi)", str(e))
    src = Path(os.environ.get("M4_SRC", Path.home() / "strata-m4-src"))
    line((src / "terminal" / "strata.py").exists(), "Strata CLI for the monitor", str(src / "terminal" / "strata.py"))
    try:
        tx = subprocess.run([str(HOME / "venv" / "bin" / "python"), "-c", "import textual; print(textual.__version__)"],
                            capture_output=True, text=True)
        line(tx.returncode == 0, "textual (monitor UI)", tx.stdout.strip() or "missing: pip install -r requirements-monitor.txt")
    except OSError:
        line(False, "textual (monitor UI)", "no venv")
    h = healthy(env)
    line(True, f"server on port {port(env)}", "running" if h else ("process up, not answering" if server_pids() else "stopped"))
    on_path = any(Path(d, "strata-m4").exists() for d in os.environ.get("PATH", "").split(":") if d)
    line(on_path, "strata-m4 on PATH", "yes" if on_path else "add ~/.local/bin to PATH (log in again, or export PATH)")
    return 0 if ok else 1


def cmd_script(a, env) -> int:
    if healthy(env) or server_pids():
        if not a.force:
            sys.exit(f"strata-m4: the server is running and holds the GPU; strata-m4 stop first (or --force to run anyway)")
    path = HOME / f"{a.command}.sh"
    if not path.exists():
        sys.exit(f"strata-m4: {path} is missing (install.sh copies it)")
    os.execv(str(path), [str(path), *a.rest])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="strata-m4", description="Mistral Small 4 on strata-m4: server, monitor, tools",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n")[1])
    sp = ap.add_subparsers(dest="command", required=True)
    p = sp.add_parser("launch", help="start the server in the background if needed, then open the monitor")
    p.add_argument("--no-monitor", action="store_true", help="only start the server")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the server (default 600)")
    p.add_argument("--grace", type=float, default=30.0)
    p = sp.add_parser("serve", help="run the server in the foreground (Ctrl+C stops it)")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="extra m4serve options")
    p = sp.add_parser("stop", help="stop the server (SIGINT, then forced after --grace seconds)")
    p.add_argument("--grace", type=float, default=30.0)
    p = sp.add_parser("restart", help="stop, then start in the background")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--grace", type=float, default=30.0)
    p = sp.add_parser("status", help="state, model, queue, rates, GPU")
    p.add_argument("--json", action="store_true")
    p = sp.add_parser("monitor", help="Strata's terminal monitor on this server")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="options of strata monitor, e.g. --once")
    p = sp.add_parser("logs", help="the server log")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", type=int, default=60)
    p = sp.add_parser("config", help="show or change ~/strata-m4/m4.env")
    p.add_argument("action", nargs="?", choices=("show", "edit", "set", "args"))
    p.add_argument("values", nargs=argparse.REMAINDER)
    sp.add_parser("doctor", help="read-only checks")
    for name in SCRIPTS:
        p = sp.add_parser(name, help=f"run {name}.sh (needs the GPU: the server must be stopped)")
        p.add_argument("--force", action="store_true", help="run even while the server is up")
        p.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    env = read_env()
    fn = {"launch": cmd_launch, "serve": cmd_serve, "stop": cmd_stop, "restart": cmd_restart, "status": cmd_status,
          "monitor": cmd_monitor, "logs": cmd_logs, "config": cmd_config, "doctor": cmd_doctor}.get(a.command, cmd_script)
    return fn(a, env)


if __name__ == "__main__":
    sys.exit(main())
