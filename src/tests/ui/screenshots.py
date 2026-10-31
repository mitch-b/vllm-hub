"""Render the hub UI with mocked API data and save screenshots.

Runs without a GPU, Docker or network: index.html is served from the repo and every
/api call is answered from tests/ui/fixtures. Used by CI to post screenshots on PRs
and to refresh the README images.

    python tests/ui/screenshots.py --out shots [--scene connect]
"""

import argparse
import json
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent
FIX = HERE / "fixtures"
INDEX = HERE.parent.parent / "app" / "static" / "index.html"
BASE = "http://hub.test"
CAPTURED = 1793462969  # when the fixtures were recorded; timestamps are shifted to now so "used 15d ago" never drifts
VIEWPORT = {"width": 1280, "height": 820}


def fixture(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def open_page(browser, overrides=None):
    data = {n: fixture(n) for n in ("system", "models", "version", "recommended", "jobs", "search", "chat")}
    data.update(overrides or {})
    shift = time.time() - CAPTURED
    for m in data["models"]:
        m["last_used"] += shift
    data["version"]["checked_at"] += shift
    startup = data["system"]["vllm"].get("startup")
    if startup:
        for key in ("started_at", "last_log_at"):
            if startup.get(key) is not None:
                startup[key] += shift
    page = browser.new_page(viewport=VIEWPORT, color_scheme="dark")

    def api(route):
        path = route.request.url.split(BASE, 1)[1].split("?")[0]
        if path == "/api/logs":
            return route.fulfill(content_type="text/plain", body=(FIX / "logs.txt").read_text(encoding="utf-8"))
        key = path.removeprefix("/api/").split("/")[0]
        if key == "chat":
            time.sleep(1.7)  # realistic duration and tok/s in the playground footer
        body = data.get(key, {})
        route.fulfill(content_type="application/json", body=json.dumps(body))

    page.route(f"{BASE}/api/**", api)
    page.route(f"{BASE}/", lambda r: r.fulfill(content_type="text/html", body=INDEX.read_text(encoding="utf-8")))
    page.goto(BASE + "/")
    page.wait_for_selector("#tab-library")
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(500)
    return page


def toggle(page, target):
    page.click(f'[data-act="toggle"][data-target="{target}"]')
    page.wait_for_timeout(300)


def overview(b):
    p = open_page(b)
    p.click("#tab-library")
    return p


def connect(b):
    p = overview(b)
    toggle(p, "connect")
    p.wait_for_selector("#cliCode")
    return p


def playground(b):
    p = overview(b)
    toggle(p, "playground")
    p.fill("#prompt", "In one sentence, what is vLLM?")
    p.keyboard.press("Enter")
    p.wait_for_selector(".msg.assistant .meta")
    return p


def run_dialog(b):
    p = overview(b)
    p.locator('[data-act="run"]:visible').last.click()
    p.wait_for_selector("dialog[open]")
    return p


def discover(b):
    p = open_page(b)
    p.click("#tab-discover")
    p.fill("#q", "qwen3 awq")
    p.wait_for_selector('[data-act="pull"]')
    p.locator("#q").blur()
    return p


def downloading(b):
    jobs = fixture("downloading")
    model_id = jobs[0]["repo_id"]
    models = fixture("models") + [{"id": model_id, "size_gib": 0, "last_used": CAPTURED}]
    search = [dict(fixture("recommended")[0], id=model_id)]
    p = open_page(b, {"jobs": jobs, "models": models, "search": search})
    p.click("#tab-library")
    p.wait_for_selector('[data-act="cancel-job"]')
    # A partial cache and older failed attempt must not add duplicate rows or a Run action.
    for status in ("queued", "downloading", "error", "cancelled", "done"):
        result = p.evaluate(
            """status => {
                S.jobs[0].status = status;
                renderAll();
                const id = S.jobs[0].repo_id;
                return {
                    rows: document.querySelectorAll('#library .item').length,
                    count: +document.querySelector('#libCount').textContent,
                    run: !!document.querySelector(`#library [data-act="run"][data-id="${id}"]`),
                    retry: !!document.querySelector('#library [data-act="pull"]'),
                    local: isLocal(id),
                };
            }""",
            status,
        )
        assert result["rows"] == result["count"] == len(models), (status, result)
        assert result["run"] == (status == "done"), (status, result)
        assert result["retry"] == (status in ("error", "cancelled")), (status, result)
        assert result["local"] == (status not in ("queued", "downloading")), (status, result)
    p.evaluate("S.jobs[0].status = 'downloading'; renderAll()")
    p.click("#tab-discover")
    p.fill("#q", model_id)
    p.wait_for_selector('#discover .progress')
    assert p.locator('#discover [data-act="run"]').count() == 0
    assert p.locator('#discover [data-act="pull"]').count() == 0
    p.click("#tab-library")
    return p


def downloading_run(b):
    p = downloading(b)
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    system = fixture("system")
    p.locator(f'#library [data-act="run"][data-id="{model_id}"]').click()
    p.wait_for_selector("#runDlg[open]")
    assert p.locator("#runGo").inner_text().strip().endswith("Run")
    requests = []

    def run(route):
        requests.append(route.request.post_data_json)
        system["vllm"]["model"] = model_id
        system["vllm"]["context"] = "32768"
        system["vllm"]["served"].update(id="qwen2.5-7b-instruct", root=model_id, max_model_len=32768)
        route.fulfill(content_type="application/json", body='{"context": 32768}')

    p.route(
        f"{BASE}/api/system",
        lambda r: r.fulfill(content_type="application/json", body=json.dumps(system)),
    )
    p.route(f"{BASE}/api/run", run)
    with p.expect_request(f"{BASE}/api/run"):
        p.click("#runGo")
    p.wait_for_selector("#runDlg[open]", state="hidden")
    p.wait_for_function("id => S.sys.vllm.ready && S.sys.vllm.model === id", arg=model_id)
    assert len(requests) == 1 and requests[0]["repo_id"] == model_id
    assert p.locator('#library [data-act="cancel-job"]').count() == 1
    assert p.locator('#library .item').count() == 4
    assert p.evaluate("S.jobs[0].status") == "downloading"
    return p


def console(b):
    p = overview(b)
    toggle(p, "console")
    return p


def starting(b):
    system = fixture("system")
    system["vllm"].update(ready=False, served=None, startup=fixture("starting"))
    p = open_page(b, {"system": system})
    p.wait_for_selector("#startupStage")
    assert p.locator("#startupStage").inner_text() == "Loading weights · 3 / 8 shards"
    assert p.locator("#elapsed").inner_text().startswith("2m ")
    p.reload()
    p.wait_for_selector("#startupStage")
    assert p.locator("#elapsed").inner_text().startswith("2m ")
    assert p.locator("#startupStage").inner_text() == "Loading weights · 3 / 8 shards"
    # The client advances server-relative durations even when its wall clock is skewed.
    p.evaluate("S.sysAt -= 10000; renderRuntime()")
    assert p.locator("#elapsed").inner_text().startswith("2m 15")
    return p


def warming(b):
    p = starting(b)
    p.evaluate("""() => {
        S.sys.vllm.startup.stage = 'Warming up';
        S.sys.vllm.startup.progress = null;
        S.sys.vllm.startup.last_log_at -= 60;
        renderRuntime();
    }""")
    assert p.locator("#startupStage").inner_text() == "Warming up"
    assert "does not mean it is stuck" in p.locator("#startupActivity").inner_text()
    # A replacement container and then a Docker restart must not inherit the old timer.
    p.evaluate("""() => {
        S.sys.vllm.model = 'fixture/new-model';
        S.sys.vllm.startup = {
            container_id: 'replacement', elapsed_seconds: 5,
            started_at: 1000, last_log_at: null, stage: 'Starting engine', progress: null
        };
        S.sysAt = Date.now(); renderRuntime();
    }""")
    assert p.locator("#elapsed").inner_text() == "5s"
    p.evaluate("S.sys.vllm.state = 'restarting'; S.sys.vllm.startup = {stage: 'Container restarting'}; renderRuntime()")
    assert p.locator("#elapsed").inner_text() == "Waiting for container"
    assert p.locator("#startupStage").inner_text() == "Container restarting"
    p.evaluate("S.sys.vllm.ready = true; renderRuntime()")
    assert p.locator("#startupStage").count() == 0
    p.evaluate("S.sys.vllm.ready = false; S.sys.vllm.state = 'exited'; renderRuntime()")
    assert p.locator("#startupStage").count() == 0
    p.evaluate("""() => {
        S.sys.vllm.state = 'running';
        S.sys.vllm.startup = {elapsed_seconds: 5, stage: 'Starting engine',
            error: 'Startup logs unavailable: logging driver does not support reading'};
        renderRuntime();
    }""")
    assert "Startup logs unavailable" in p.locator("#startupActivity").inner_text()
    p.reload()
    p.wait_for_selector("#startupStage")
    p.evaluate("""() => {
        S.sys.vllm.startup.stage = 'Warming up';
        S.sys.vllm.startup.progress = null;
        S.sys.vllm.startup.last_log_at -= 60;
        renderRuntime();
    }""")
    return p


def capturing(b):
    p = starting(b)
    p.evaluate("""() => {
        S.sys.vllm.startup.stage = 'Capturing CUDA graphs';
        S.sys.vllm.startup.progress = {completed: 12, total: 24};
        renderRuntime();
    }""")
    assert p.locator("#startupStage").inner_text() == "Capturing CUDA graphs · 12 / 24 graphs"
    return p


def update(b):
    v = fixture("version")
    v["update_available"] = True
    v["remote"] = dict(
        v["remote"],
        version="20261101-0900",
        revision="f" * 40,
        digest="sha256:" + "f" * 64,
        created="2026-11-01T14:00:00Z",
    )
    p = open_page(b, {"version": v})
    p.click("#ver")
    p.wait_for_selector("#updDlg[open]")
    return p


SCENES = {
    f.__name__.replace("_dialog", ""): f
    for f in (overview, connect, playground, run_dialog, discover, downloading, downloading_run,
              console, starting, warming, capturing, update)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="shots")
    ap.add_argument("--scene", action="append", choices=sorted(SCENES))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for name in args.scene or SCENES:
            try:
                page = SCENES[name](browser)
                page.wait_for_timeout(400)
                page.screenshot(path=out / f"{name}.png")
                page.close()
                print(f"ok   {name}")
            except Exception as e:  # keep going so one broken scene doesn't hide the others
                failed.append(name)
                print(f"FAIL {name}: {e}", file=sys.stderr)
        browser.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
