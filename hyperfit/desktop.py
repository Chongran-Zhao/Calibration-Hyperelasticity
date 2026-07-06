"""Desktop launcher: local FastAPI server + native pywebview window.

Usage::

    hyperfit-app             # open the native desktop window
    hyperfit-app --browser   # serve and open the default web browser instead
    hyperfit-app --check     # headless smoke test (start, probe, exit)

The server binds to 127.0.0.1 on a free port, serves both the JSON API and
the built frontend, and shuts down when the window closes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import socket
import sys
import threading
import time
import urllib.request

APP_TITLE = "Hyperelastic Calibration"
WINDOW_SIZE = (1440, 900)

logger = logging.getLogger(__name__)


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server(port: int):
    """Run uvicorn in a daemon thread; returns the server handle."""
    import uvicorn

    from .api import create_app

    config = uvicorn.Config(
        create_app(),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="hyperfit-uvicorn", daemon=True)
    thread.start()
    return server


def _wait_for_health(base_url: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1.0) as response:
                return json.load(response)
        except Exception as exc:  # noqa: BLE001 - retry until deadline
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Server did not become healthy at {base_url}: {last_error}")


def _check(base_url: str) -> int:
    """Headless smoke test used by CI and packaging."""
    health = _wait_for_health(base_url)
    print(f"health: {health}")
    with urllib.request.urlopen(base_url, timeout=5.0) as response:
        index = response.read()
    ok_index = b"<div id=\"root\"" in index or b"<div id='root'" in index
    print(f"frontend index served: {len(index)} bytes, root div: {ok_index}")
    with urllib.request.urlopen(f"{base_url}/api/datasets", timeout=10.0) as response:
        datasets = json.load(response)
    print(f"datasets: {len(datasets.get('authors', []))} authors")
    if not (health.get("ok") and ok_index and datasets.get("authors")):
        print("SMOKE TEST FAILED")
        return 1
    print("SMOKE TEST PASSED")
    return 0


def _serve_browser(base_url: str) -> int:
    """Open the default browser and keep serving until interrupted."""
    import webbrowser

    webbrowser.open(base_url)
    print(f"{APP_TITLE} running at {base_url} (Ctrl-C to quit)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hyperfit-app", description=__doc__)
    parser.add_argument("--browser", action="store_true", help="open in the default browser instead of a native window")
    parser.add_argument("--check", action="store_true", help="headless smoke test: start server, probe endpoints, exit")
    parser.add_argument("--port", type=int, default=None, help="fixed port (default: auto)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = _start_server(port)

    try:
        if args.check:
            return _check(base_url)

        _wait_for_health(base_url)
        logger.info("Serving %s at %s", APP_TITLE, base_url)

        if args.browser:
            return _serve_browser(base_url)

        try:
            import webview
        except ImportError:
            print(
                "pywebview is not installed; falling back to the browser.\n"
                "Install the native window with: pip install 'hyperfit[desktop]'",
                file=sys.stderr,
            )
            return _serve_browser(base_url)

        webview.create_window(
            APP_TITLE,
            base_url,
            width=WINDOW_SIZE[0],
            height=WINDOW_SIZE[1],
            min_size=(1100, 700),
        )
        webview.start()
        return 0
    finally:
        server.should_exit = True


if __name__ == "__main__":
    sys.exit(main())
