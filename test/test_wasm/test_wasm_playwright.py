
from tempfile import TemporaryDirectory
import shutil
import socket
import threading
from contextlib import closing, contextmanager
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
WORK_DIR = os.path.join(THIS_DIR, "work_dir")


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def start_server(work_dir, port):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=work_dir, **kwargs)

        def log_message(self, fmt, *args):
            return

    httpd = HTTPServer(("127.0.0.1", port), Handler)

    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    return thread, httpd


@contextmanager
def server_context(work_dir, port):
    thread, server = start_server(work_dir=work_dir, port=port)
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join()

async def playwright_run_page(page_url, headless=True, slow_mo=None):
    async with async_playwright() as p:
        if slow_mo is None:
            browser = await p.chromium.launch(headless=headless)
        else:
            browser = await p.chromium.launch(
                headless=headless, slow_mo=slow_mo
            )
        page = await browser.new_page()
        await page.goto(page_url)
        # n min = n_min * 60 * 1000 ms
        n_min = 4
        page.set_default_timeout(n_min * 60 * 1000)

        async def handle_console(msg):
            txt = str(msg)
            print(txt)

        page.on("console", handle_console)


        status = await page.evaluate(
            f"""async () => {{
                let test_module = await test_xsimd_wasm();
                console.log("\\n\\n************************************************************");
                console.log("XSIMD WASM TESTS:");
                console.log("************************************************************");
                let r = test_module.run_tests();
                if (r == 0) {{
                console.log("\\n\\n************************************************************");
                console.log("XSIMD WASM TESTS PASSED");
                console.log("************************************************************");
                return r;
                }}
                else {{
                console.log("************************************************************");
                console.log("XSIMD WASM TESTS FAILED");
                console.log("************************************************************");
                return r;
                }}

                }}"""
        )
        return_code = int(status)
    return return_code
def main(build_dir):
    
    work_dir = WORK_DIR# TemporaryDirectory()

    with TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)


        shutil.copy(f"{build_dir}/test_xsimd.wasm", work_dir)
        shutil.copy(f"{build_dir}/test_xsimd.js", work_dir)
        shutil.copy(f"{THIS_DIR}/browser_main.html", work_dir)
        
        port = find_free_port()
        with server_context(work_dir=work_dir, port=port) as (server, url):
            page_url = f"{url}/browser_main.html"
            ret = asyncio.run(playwright_run_page(page_url=page_url))
        
            return ret



if __name__ == "__main__":
    import sys

    # get arg from args
    build_dir = sys.argv[1]
    
    print(f"build_dir: {build_dir}")

    ret_code = main(build_dir)
    sys.exit(ret_code)