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


def console(b):
    p = overview(b)
    toggle(p, "console")
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
    f.__name__.replace("_dialog", ""): f for f in (overview, connect, playground, run_dialog, discover, console, update)
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
