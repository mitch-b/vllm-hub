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


def open_page(browser, overrides=None, *, platform=None):
    data = {n: fixture(n) for n in ("system", "models", "version", "recommended", "jobs", "search", "chat", "activity")}
    data.update(overrides or {})
    shift = time.time() - CAPTURED
    for m in data["models"]:
        if m["last_used"]:
            m["last_used"] += shift
    data["version"]["checked_at"] += shift
    act = data["activity"]
    if act.get("serving"):
        act["now"] += shift
        act["since"] += shift
        for row in act["history"] + act["recent"]:
            row["t"] += shift
        for c in act["callers"]:
            c["first_seen"] += shift
            c["last_seen"] += shift
    startup = data["system"]["vllm"].get("startup")
    if startup:
        for key in ("started_at", "last_log_at"):
            if startup.get(key) is not None:
                startup[key] += shift
    page = browser.new_page(viewport=VIEWPORT, color_scheme="dark")
    if platform:
        page.add_init_script(
            "Object.defineProperty(navigator, 'userAgentData', {value: "
            + json.dumps({"platform": platform}) + "});"
        )

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
    for platform, shell in (("Windows", "pwsh"), ("macOS", "bash"), ("Linux", "bash")):
        page = open_page(b, platform=platform)
        toggle(page, "connect")
        page.wait_for_selector("#cliCode")
        assert page.evaluate("conn.shell") == shell
        page.evaluate("conn.shell = 'cmd'; saveConn()")
        page.reload()
        page.wait_for_selector("#tab-library")
        assert page.evaluate("conn.shell") == "cmd"
        page.close()
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


def download_finalizing(b):
    jobs = fixture("downloading")
    jobs[0].update(done=jobs[0]["total"], percent=99.9, finalizing=True, speed_bps=0, eta_s=None)
    model_id = jobs[0]["repo_id"]
    models = fixture("models") + [{"id": model_id, "size_gib": 0, "last_used": CAPTURED}]
    p = open_page(b, {"jobs": jobs, "models": models, "search": [dict(fixture("recommended")[0], id=model_id)]})
    p.click("#tab-library")
    p.wait_for_selector('#library .meta:has-text("Finalizing snapshot")')
    assert "100%" not in p.locator("#library").inner_text()
    assert p.locator(f'#library [data-act="run"][data-id="{model_id}"]').count() == 0
    assert p.locator('#library .item').count() == len(models)
    p.click("#tab-discover")
    p.fill("#q", model_id)
    p.wait_for_selector('#discover .badge:has-text("Finalizing snapshot")')
    assert p.locator('#discover [data-act="run"]').count() == 0
    p.click("#tab-library")
    # Rounding a near-complete progress value must not display 100% either.
    p.evaluate("S.jobs[0].finalizing = false; renderAll()")
    assert "99%" in p.locator("#library").inner_text()
    assert "100%" not in p.locator("#library").inner_text()
    p.evaluate("S.jobs[0].finalizing = true; renderAll()")
    return p


def download_completion(b):
    p = download_finalizing(b)
    job = dict(fixture("downloading")[0], status="done", percent=100, finalizing=False)
    model_id = job["repo_id"]
    models = fixture("models") + [{"id": model_id, "size_gib": 16.9, "last_used": CAPTURED}]
    p.route(f"{BASE}/api/jobs", lambda r: r.fulfill(content_type="application/json", body=json.dumps([job])))
    responses = ["error", "stale", "ready"]

    def cache(route):
        response = responses.pop(0) if responses else "ready"
        route.fulfill(
            status=503 if response == "error" else 200,
            content_type="application/json",
            body=json.dumps({"detail": "Cache temporarily unavailable"} if response == "error"
                            else models if response == "ready" else []),
        )

    p.route(f"{BASE}/api/models", cache)
    # First observation is already done: reload/missed polling must still refresh.
    p.evaluate("S.prevJob = {}; S.local = []; refreshJobs()")
    p.wait_for_selector('#library .meta:has-text("Updating library")')
    assert p.locator('#library [data-act="run"]').count() == 0
    assert p.evaluate("S.prevJob.download1") is None
    assert "Could not refresh library" in p.locator("#toasts").inner_text()
    p.evaluate("refreshJobs()")
    assert p.locator('#library [data-act="run"]').count() == 0
    p.evaluate("refreshJobs()")
    p.wait_for_selector(f'#library [data-act="run"][data-id="{model_id}"]')
    assert p.locator('#library .item').count() == len(models)
    assert p.evaluate("S.prevJob.download1") == "done"
    assert "Download complete" not in p.locator("#toasts").inner_text()

    # A stale /models response arriving after completion must recover on the next poll.
    p.evaluate("S.local = []; refreshJobs()")
    p.wait_for_selector(f'#library [data-act="run"][data-id="{model_id}"]')
    requests = []
    p.route(
        f"{BASE}/api/run",
        lambda r: (requests.append(r.request.post_data_json),
                   r.fulfill(content_type="application/json", body='{"context":32768}')),
    )
    p.route(f"{BASE}/api/pull", lambda r: r.fulfill(content_type="application/json", body=json.dumps(job)))
    # /pull itself can return done before the first /jobs poll (fully cached / fast job).
    with p.expect_request(f"{BASE}/api/run"):
        p.evaluate("id => pull(id, {id, context:32768, graphs:false, util:0.9, w:16.9})", model_id)
    p.evaluate("refreshJobs()")
    assert len(requests) == 1 and requests[0]["repo_id"] == model_id
    assert p.evaluate("Object.keys(S.runAfter).length") == 0
    # Actual reload: the initial cache request predates completion, the first job poll is done.
    responses[:] = ["stale", "ready"]
    p.reload()
    p.click("#tab-library")
    p.wait_for_selector(f'#library [data-act="run"][data-id="{model_id}"]')
    assert p.locator('#library .item').count() == len(models)
    assert len(requests) == 1
    return p


def console(b):
    p = overview(b)
    toggle(p, "console")
    return p


def activity(b):
    p = overview(b)
    toggle(p, "activity")
    p.wait_for_selector("#actBody svg.spark path.line")
    text = p.locator("#actBody").inner_text()
    assert "This computer" in text and "412 requests served" in text, text
    # Only offered while serving: a starting model hides the toggle and the drawer.
    p.evaluate("S.sys.vllm.ready = false; renderRuntime()")
    assert p.locator('#rtMain [data-target="activity"]').count() == 0
    assert p.locator("#activity").is_hidden()
    p.evaluate("S.sys.vllm.ready = true; renderRuntime(); loadActivity()")
    p.wait_for_selector("#activity", state="visible")
    p.wait_for_selector("#actBody svg.spark path.line")
    p.locator("#activity").scroll_into_view_if_needed()
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


def cancelled_start(b):
    p = starting(b)
    system = fixture("system")
    system["vllm"] = {"docker_ok": True, "state": "stopped", "model": None, "ready": False}
    p.route(f"{BASE}/api/system", lambda r: r.fulfill(
        content_type="application/json", body=json.dumps(system)))
    p.route(f"{BASE}/api/stop", lambda r: r.fulfill(
        content_type="application/json", body='{"ok":true}'))
    p.evaluate("""() => {
        const original = window.fetch;
        window.runCalls = 0;
        window.fetch = (url, opts) => {
            if (url === '/api/run') {
                ++window.runCalls;
                return new Promise(resolve => {
                    window.finishRun = () => resolve(new Response('{"context":8192}'));
                });
            }
            return original(url, opts);
        };
        localStorage.removeItem('hub.last');
        S.runAfter.download1 = {id:'fixture/queued-model', context:8192};
        window.pendingRun = startModel({id:'fixture/loading-model', context:8192});
    }""")
    p.locator('#rtMain [data-act="stop"]').click()
    p.wait_for_function("phase() === 'idle'")
    assert p.evaluate("Object.keys(S.runAfter).length") == 0
    p.evaluate("finishRun(); pendingRun")
    assert p.evaluate("store.get('last', null)") is None
    assert p.evaluate("window.runCalls") == 1
    assert "Could not start vLLM" not in p.locator("#toasts").inner_text()
    job = dict(fixture("downloading")[0], status="done", percent=100, finalizing=False)
    models = fixture("models") + [{"id": job["repo_id"], "size_gib": 16.9, "last_used": CAPTURED}]
    p.route(f"{BASE}/api/jobs", lambda r: r.fulfill(
        content_type="application/json", body=json.dumps([job])))
    p.route(f"{BASE}/api/models", lambda r: r.fulfill(
        content_type="application/json", body=json.dumps(models)))
    p.evaluate("refreshJobs()")
    assert p.evaluate("window.runCalls") == 1
    assert p.locator("#rtMain").inner_text().startswith("No model running")
    return p


def console_recovery(b):
    p = starting(b)
    toggle(p, "console")
    p.wait_for_function("document.querySelector('#logs').textContent.includes('INFO')")
    responses = ["empty", "error", "recovered"]
    calls = []

    def logs(route):
        response = responses.pop(0) if responses else "recovered"
        calls.append(response)
        route.fulfill(
            status=503 if response == "error" else 200, content_type="text/plain",
            body="" if response == "empty" else "Logging unavailable" if response == "error"
            else "INFO Loading weights: 1/4\n",
        )

    p.route(f"{BASE}/api/logs", logs)
    p.evaluate("streamLogs()")
    assert "No console output yet" in p.locator("#logs").inner_text()
    p.evaluate("refreshSystem()")
    p.wait_for_function("document.querySelector('#logs').textContent.includes('Console disconnected')")
    assert "503" in p.locator("#logs").inner_text()
    p.evaluate("refreshSystem()")
    p.wait_for_function("document.querySelector('#logs').textContent.includes('Loading weights: 1/4')")
    assert calls[:3] == ["empty", "error", "recovered"]
    # A same-model replacement must abort the old stream and attach to the new container.
    p.evaluate("""() => {
        const original = window.fetch;
        window.fetch = (url, opts) => url === '/api/logs'
            ? Promise.resolve(new Response(new ReadableStream({
                start(controller) {
                    opts.signal.addEventListener('abort', () => {
                        controller.error(new DOMException('Aborted', 'AbortError'));
                    });
                }
            }))) : original(url, opts);
        streamLogs();
        window.fetch = original;
    }""")
    system = fixture("system")
    system["vllm"].update(ready=False, served=None, container_id="replacement", startup=fixture("starting"))
    p.route(f"{BASE}/api/system", lambda r: r.fulfill(
        content_type="application/json", body=json.dumps(system)))
    p.evaluate("refreshSystem()")
    p.wait_for_function("document.querySelector('#logs').textContent.includes('Loading weights: 1/4')")
    assert p.evaluate("S.logsFor.startsWith('replacement|')")
    assert "Console disconnected" not in p.locator("#logs").inner_text()
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


def hosting_page(b, environment):
    system = fixture("system")
    system["gpu"] = None
    system["recommended_util"] = None
    system["hosting"] = {"local_inference": fixture("hosting")[environment]}
    system["vllm"] = {
        "docker_ok": environment != "no_docker",
        "state": "stopped", "model": None, "ready": False,
    }
    p = open_page(b, {"system": system})
    p.click("#tab-library")
    p.wait_for_selector('#library [data-act="run"]:disabled')
    assert p.locator('#library [data-act="run"]:enabled').count() == 0
    p.locator('#library [data-act="menu-local"]').first.click()
    assert "Run with options" not in p.locator(".menu").inner_text()
    p.keyboard.press("Escape")
    p.click("#tab-discover")
    p.wait_for_selector('#discover [data-act="pull"]:enabled')
    assert p.locator('#discover [data-act="run"]:enabled').count() == 0
    assert "Model catalog" in p.locator("#discLabel").text_content()
    assert p.locator('#discover .badge:has-text("Fits")').count() == 0
    assert p.locator("#showAll").is_disabled()
    requests = []
    p.route(f"{BASE}/api/run", lambda r: (
        requests.append(r.request.post_data_json),
        r.fulfill(content_type="application/json", body="{}"),
    ))
    p.evaluate("startModel({id:'fixture/model'}); openRunDialog('fixture/model', 4)")
    assert p.locator("#runDlg[open]").count() == 0
    assert requests == []
    job = dict(fixture("downloading")[0], status="queued")
    p.route(f"{BASE}/api/pull", lambda r: r.fulfill(
        content_type="application/json", body=json.dumps(job)))
    with p.expect_request(f"{BASE}/api/pull"):
        p.locator('#discover [data-act="pull"]:enabled').first.click()
    assert p.evaluate("Object.keys(S.runAfter).length") == 0
    # Capability recovery re-enables the same UI without a reload.
    p.evaluate("""support => {
        window.unavailableSystem = S.sys;
        S.sys = support;
        renderAll();
    }""", fixture("system"))
    assert p.locator('#library [data-act="run"]:enabled').count() > 0
    p.evaluate("S.runAfter.pending = {id:'fixture/model'}; refreshSystem()")
    assert p.evaluate("Object.keys(S.runAfter).length") == 0
    assert "Queued starts cancelled" in p.locator("#toasts").inner_text()
    p.evaluate("S.tab = 'library'; renderTabs(); $('#toasts').replaceChildren()")
    return p


def macos(b):
    data = fixture("native")
    system = data["system"]
    p = open_page(b, {**data, "jobs": []})
    p.click("#tab-library")
    p.wait_for_selector('#library [data-act="run"]:enabled')
    assert "Model Runner" in p.locator("#gpu").inner_text()
    p.locator('#library [data-act="run"]').first.click()
    p.wait_for_selector("#runDlg[open]")
    assert p.locator("#runDlg details").is_hidden()
    assert p.locator('#ctxSeg [data-v="0"]').get_attribute("aria-checked") == "true"
    requests = []

    def start(route):
        request = route.request.post_data_json
        requests.append(request)
        system["vllm"].update(
            state="running", ready=True, model=request["repo_id"], context=8192,
            engine="llama.cpp", served={"id": request["repo_id"], "max_model_len": 8192},
        )
        route.fulfill(content_type="application/json", body='{"context":8192,"util":null,"args":[]}')

    p.route(f"{BASE}/api/system", lambda r: r.fulfill(
        content_type="application/json", body=json.dumps(system)))
    p.route(f"{BASE}/api/run", start)
    with p.expect_request(f"{BASE}/api/run"):
        p.click("#runGo")
    p.wait_for_function("S.sys.vllm.ready")
    assert requests[0]["graphs"] is False and requests[0]["util"] is None
    assert requests[0]["context"] is None
    assert p.locator('#rtMain [data-target="activity"]').count() == 0
    assert p.locator('#rtMain [data-target="console"]').count() == 0
    toggle(p, "connect")
    p.wait_for_selector("#cliCode")
    assert "http://localhost:12434/engines/v1" in p.locator("#cliCode").inner_text()
    assert "ai/qwen3:0.6B-Q4_K_M" in p.locator("#cliCode").inner_text()
    assert p.evaluate("conn.shell") == "bash"
    assert "COPILOT_MODEL=ai/qwen3:0.6B-Q4_K_M" in p.locator("#cliCode").inner_text()
    limits = p.evaluate("tokenLimits()")
    assert limits["prompt"] + limits["out"] == 8192
    toggle(p, "connect")
    return p


def macos_mobile(b):
    p = macos(b)
    p.set_viewport_size({"width": 390, "height": 844})
    assert p.evaluate("document.documentElement.scrollWidth <= innerWidth")
    return p


def macos_start_failed(b):
    data = fixture("native")
    data["system"]["vllm"].update(
        state="exited", model=data["models"][0]["id"], context=8192,
        error="Native engine could not load the model. Check Docker Desktop Models for details.",
    )
    p = open_page(b, {**data, "jobs": []})
    assert "Start failed" in p.locator("#rtMain").inner_text()
    assert "Native engine could not load" in p.locator("#rtMain").inner_text()
    assert p.locator('#rtMain [data-target="console"]').count() == 0
    assert p.locator('#rtMain [data-act="restart"]').is_enabled()
    return p


def macos_connection_lost(b):
    p = macos(b)
    serving = p.evaluate("S.sys")

    def disconnected(route):
        route.fulfill(status=502, content_type="application/json",
                      body='{"detail":"Native management connection lost"}')

    p.route(f"{BASE}/api/system", disconnected)
    p.evaluate("S.runAfter.pending = {id: 'fixture/model'}; refreshSystem()")
    assert "Inference status unavailable" in p.locator("#rtMain").inner_text()
    assert "Native management connection lost" in p.locator("#rtMain").inner_text()
    assert "Serving" not in p.locator("#rtMain").inner_text()
    assert p.locator('#library [data-act="run"]:enabled').count() == 0
    assert p.evaluate("Object.keys(S.runAfter).length") == 0
    p.route(f"{BASE}/api/system", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(serving)))
    p.evaluate("refreshSystem()")
    assert "Serving" in p.locator("#rtMain").inner_text()
    p.route(f"{BASE}/api/system", disconnected)
    p.evaluate("refreshSystem()")
    return p


def no_gpu(b):
    p = hosting_page(b, "no_gpu")
    assert "Set up local inference" in p.locator("#rtMain").inner_text()
    return p


def no_docker(b):
    p = hosting_page(b, "no_docker")
    assert "Docker is unreachable" in p.locator("#rtMain").inner_text()
    return p


SCENES = {
    f.__name__.replace("_dialog", ""): f
    for f in (overview, connect, playground, activity, run_dialog, discover, downloading, downloading_run,
              download_finalizing, download_completion, console, starting, warming, capturing,
              cancelled_start, console_recovery, update, macos, macos_mobile, macos_start_failed,
              macos_connection_lost, no_gpu, no_docker)
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
