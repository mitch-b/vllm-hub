"""Live caller activity for the served model, from vLLM's Prometheus /metrics and its HTTP access log.

Sampled in the background while a model is serving and kept in memory only (15 minutes).
Long-term dashboards can scrape vLLM's /metrics directly with Prometheus.
"""
from __future__ import annotations

import os
import re
import socket
import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock

import httpx

from . import runner

INTERVAL = float(os.getenv("HUB_ACTIVITY_INTERVAL", "5"))
WINDOW = 15 * 60
RECENT = 40
LOG_TAIL = 2000
METRIC_RE = re.compile(r"^(vllm:[A-Za-z0-9_]+)(?:\{[^}]*\})?\s+(\S+)")
ACCESS_RE = re.compile(r'(?P<ip>\[[0-9a-fA-F:.]+\]|[0-9a-fA-F:.]+):\d+ - '
                       r'"(?P<method>[A-Z]+) (?P<path>\S+) HTTP/[\d.]+" (?P<status>\d{3})')
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z$")
# Health checks, scrapes and the hub's own status polling are not caller traffic.
NOISE = {"/metrics", "/health", "/ping", "/version", "/v1/models", "/load", "/is_sleeping", "/favicon.ico"}

_lock = Lock()


class _State:
    def __init__(self, key: str | None = None, started: float | None = None, labels: dict | None = None):
        self.key, self.started, self.labels = key, started, labels or {}
        self.serving = False
        self.sampled_at = 0.0
        self.prev: tuple[float, dict] | None = None
        self.metrics: dict = {}
        self.samples: deque = deque()
        self.callers: dict[str, dict] = {}
        self.recent: deque = deque(maxlen=RECENT)
        self.log_since = started or 0.0


_state = _State()


def reset() -> None:
    global _state
    with _lock:
        _state = _State()


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each vllm:* series across its label sets (engines, finish reasons)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = METRIC_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(2))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + value
    return out


def _ts(value: str) -> float | None:
    m = TS_RE.match(value)
    if not m:
        return None
    base = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    return base + (float("0." + m.group(2)) if m.group(2) else 0.0)


def _own_ips() -> set[str]:
    try:
        return set(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        return set()


def _gateways() -> set[str]:
    """Requests through the published port reach vLLM from the Docker network's gateway."""
    try:
        cfg = runner.client().networks.get(runner.NETWORK).attrs.get("IPAM", {}).get("Config") or []
    except Exception:
        return set()
    return {c["Gateway"] for c in cfg if c.get("Gateway")}


def _labels() -> dict[str, str]:
    labels = {ip: "This computer" for ip in _gateways()}
    labels.update({ip: "vLLM Hub" for ip in _own_ips()})
    return labels


def _rate(cur: dict, old: dict, name: str, dt: float) -> float:
    return max(0.0, cur.get(name, 0.0) - old.get(name, 0.0)) / dt


def _add_metrics(s: _State, now: float, m: dict) -> None:
    s.serving, s.sampled_at, s.metrics = True, now, m
    if s.prev:
        t0, old = s.prev
        dt = now - t0
        if m.get("vllm:generation_tokens_total", 0) < old.get("vllm:generation_tokens_total", 0):
            s.samples.clear()  # the engine restarted inside the same container
        elif dt > 0:
            s.samples.append({
                "t": round(now, 1),
                "prompt_tps": round(_rate(m, old, "vllm:prompt_tokens_total", dt), 1),
                "gen_tps": round(_rate(m, old, "vllm:generation_tokens_total", dt), 1),
                "running": int(m.get("vllm:num_requests_running", 0)),
                "waiting": int(m.get("vllm:num_requests_waiting", 0)),
                "kv_cache": round(_kv(m), 4),
            })
    s.prev = (now, m)
    while s.samples and s.samples[0]["t"] < now - WINDOW:
        s.samples.popleft()


def _kv(m: dict) -> float:
    return m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc", 0.0))


def _add_requests(s: _State, entries: list[dict]) -> None:
    for e in entries:
        c = s.callers.setdefault(e["client"], {
            "client": e["client"], "label": e["label"], "requests": 0, "errors": 0, "first_seen": e["t"],
        })
        c["requests"] += 1
        c["errors"] += e["status"] >= 400
        c["last_seen"], c["last_path"] = e["t"], e["path"]
        s.recent.append(e)


def parse_access(text: str, since: float, labels: dict[str, str]) -> tuple[list[dict], float]:
    """Caller requests from `docker logs --timestamps` output newer than `since`."""
    entries, last = [], since
    for line in text.splitlines():
        stamp, _, message = line.partition(" ")
        t = _ts(stamp)
        if t is None or t <= since:
            continue
        last = max(last, t)
        m = ACCESS_RE.search(message)
        if not m:
            continue
        path = m["path"].split("?")[0]
        if path in NOISE:
            continue
        ip = m["ip"].strip("[]")
        entries.append({"t": round(t, 3), "client": ip, "label": labels.get(ip), "method": m["method"],
                        "path": path, "status": int(m["status"])})
    return entries, last


def sample(now: float | None = None) -> None:
    """Take one metrics + access-log sample of the serving model, or clear state when none is serving."""
    global _state
    now = now or time.time()
    try:
        c = runner._container()
    except Exception:
        c = None
    if c is None or c.status != "running":
        reset()
        return
    started_raw = c.attrs.get("State", {}).get("StartedAt", "")
    key = f"{c.id}|{started_raw}"
    try:
        r = httpx.get(f"{runner.BASE_URL}/metrics", timeout=3)
        r.raise_for_status()
        metrics = parse_metrics(r.text)
    except Exception:
        metrics = {}
    if not metrics:  # still loading, or not answering
        with _lock:
            _state.serving = False
        return
    with _lock:
        fresh = _state.key != key
    labels = _labels() if fresh else None
    with _lock:
        if _state.key != key:
            _state = _State(key, runner._timestamp(started_raw), labels)
        _add_metrics(_state, now, metrics)
        since, labels = _state.log_since, _state.labels
    try:
        raw = c.logs(timestamps=True, since=since or None, tail=LOG_TAIL)
        text = raw.decode(errors="replace")[-1048576:]
    except Exception:
        return
    entries, last = parse_access(text, since, labels)
    with _lock:
        if _state.key == key and _state.log_since == since:
            _state.log_since = last
            _add_requests(_state, entries)


def loop() -> None:
    while True:
        try:
            sample()
        except Exception:
            pass
        time.sleep(INTERVAL)


def snapshot() -> dict:
    with _lock:
        s = _state
        now = time.time()
        if not s.serving or now - s.sampled_at > INTERVAL * 3:
            return {"serving": False, "interval": INTERVAL}
        m = s.metrics
        last = s.samples[-1] if s.samples else {}

        def avg(name):
            n = m.get(f"{name}_count", 0)
            return round(m.get(f"{name}_sum", 0) / n, 3) if n else None

        queries = m.get("vllm:prefix_cache_queries_total", 0)
        return {
            "serving": True, "interval": INTERVAL, "window": WINDOW, "now": now, "since": s.started,
            "current": {
                "running": int(m.get("vllm:num_requests_running", 0)),
                "waiting": int(m.get("vllm:num_requests_waiting", 0)),
                "kv_cache": round(_kv(m), 4),
                "gen_tps": last.get("gen_tps", 0.0),
                "prompt_tps": last.get("prompt_tps", 0.0),
            },
            "totals": {
                "requests": int(m.get("vllm:request_success_total", 0)),
                "prompt_tokens": int(m.get("vllm:prompt_tokens_total", 0)),
                "generation_tokens": int(m.get("vllm:generation_tokens_total", 0)),
                "avg_ttft_s": avg("vllm:time_to_first_token_seconds"),
                "avg_latency_s": avg("vllm:e2e_request_latency_seconds"),
                "prefix_cache_hit": round(m.get("vllm:prefix_cache_hits_total", 0) / queries, 4) if queries else None,
            },
            "history": list(s.samples),
            "callers": sorted(s.callers.values(), key=lambda c: -c["last_seen"]),
            "recent": list(reversed(s.recent)),
            "metrics_url": f"http://localhost:{runner.HOST_PORT}/metrics",
        }
