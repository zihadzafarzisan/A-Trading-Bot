"""Local HTTP server for serving the dashboard.

Provides a simple HTTP server with optional auto-refresh capability
for viewing the generated dashboard in a browser.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger

logger = get_logger("dashboard.server")


class DashboardServer:
    """Simple HTTP server for serving the dashboard locally."""

    def __init__(
        self,
        port: int = 8000,
        directory: Optional[Path] = None,
        open_browser: bool = True,
        auto_refresh: bool = False,
    ):
        self.port = port
        # Resolve to an absolute path up front so directory resolution no longer
        # depends on start()'s os.chdir (which would make relative paths wrong).
        self.directory = Path(directory).resolve() if directory is not None else Path.cwd()
        self.open_browser = open_browser
        self.auto_refresh = auto_refresh
        self._handler: Optional[type] = None

    def start(self) -> None:
        """Start the HTTP server and optionally open browser."""
        # Change to the dashboard directory
        import os
        original_dir = os.getcwd()
        os.chdir(self.directory)

        try:
            # Create handler with optional auto-refresh
            handler_class = self._create_handler()

            with socketserver.TCPServer(("", self.port), handler_class) as httpd:
                url = f"http://127.0.0.1:{self.port}"
                logger.info(f"Serving dashboard at {url}")
                logger.info(f"Directory: {self.directory}")
                logger.info("Press Ctrl+C to stop the server")

                if self.open_browser:
                    webbrowser.open(url)

                httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nServer stopped by user")
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.warning(f"Port {self.port} is in use, trying {self.port + 1}")
                self.port += 1
                self.start()
            else:
                raise
        finally:
            os.chdir(original_dir)

    def _create_handler(self):
        """Create a custom handler with optional auto-refresh."""
        directory = self.directory
        auto_refresh = self.auto_refresh

        # Resolve a root entry-point file that actually exists in the served
        # directory. SimpleHTTPRequestHandler maps "/" to "/index.html" by
        # default, which 404s when no index.html was generated. Prefer an
        # existing index.html, otherwise fall back to a dashboard report so that
        # `serve dashboard` shows the latest generated report at the root URL.
        html_files: list[str] = []
        if directory is not None and Path(directory).is_dir():
            for _name in ("index.html", "report.html", "full_report.html"):
                if (Path(directory) / _name).is_file():
                    html_files.append(_name)
            for _f in sorted(Path(directory).glob("*.html")):
                if _f.name not in html_files:
                    html_files.append(_f.name)
        # Last resort: keep the conventional index.html name (may 404 if absent).
        root_entry = html_files[0] if html_files else "index.html"

        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            """Custom handler that can inject auto-refresh."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(directory), **kwargs)

            def end_headers(self):
                """Add cache control headers."""
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                super().end_headers()

            def translate_path(self, path):
                """Translate URL path to file path."""
                # Handle root path by serving an existing dashboard entry file.
                if path in ("/", ""):
                    path = "/" + root_entry
                return super().translate_path(path)

        if auto_refresh:
            # Wrap to inject meta refresh
            original_send_response = DashboardHandler.send_response

            def send_response_with_refresh(self, code, message=None):
                original_send_response(code, message)
                if code == 200:
                    # Check if it's an HTML file
                    path = self.path
                    if path.endswith(".html") or path == "/":
                        self.send_header("Refresh", "30")  # Refresh every 30 seconds

            DashboardHandler.send_response = send_response_with_refresh

        return DashboardHandler


def serve_dashboard(
    path: str | Path,
    port: int = 8000,
    open_browser: bool = True,
    auto_refresh: bool = False,
) -> None:
    """Serve a dashboard file or directory.

    Args:
        path: Path to the HTML file or directory containing it
        port: Port to serve on (default 8000)
        open_browser: Whether to open browser automatically
        auto_refresh: Whether to auto-refresh every 30 seconds
    """
    path = Path(path)

    if path.is_file():
        server = DashboardServer(
            port=port,
            directory=path.parent,
            open_browser=open_browser,
            auto_refresh=auto_refresh,
        )
        logger.info(f"Serving dashboard: {path}")
    else:
        server = DashboardServer(
            port=port,
            directory=path,
            open_browser=open_browser,
            auto_refresh=auto_refresh,
        )
        logger.info(f"Serving directory: {path}")

    server.start()