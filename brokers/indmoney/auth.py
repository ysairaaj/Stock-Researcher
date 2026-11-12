"""
OAuth 2.1 (PKCE) for the INDmoney MCP server.

The MCP SDK's OAuthClientProvider does discovery, dynamic client registration,
the authorization-code exchange and token refresh. This module supplies the
three pieces it needs from us:

- a TokenStorage that persists tokens AND client_info to disk, so registration
  happens once and refreshes are silent on later runs;
- a redirect handler that opens the INDmoney sign-in page in the browser;
- a callback handler: a tiny local HTTP server that catches the redirect to
  http://localhost:<port>/callback and extracts code/state.

The only secret on disk is memory/indmoney_oauth.json (gitignored).
"""
import asyncio
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from pydantic import AnyUrl

from mcp.client.auth import OAuthClientProvider, AuthorizationCodeResult
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken


MCP_URL = "https://mcp.indmoney.com/mcp"
# INDmoney's protected-resource metadata declares the resource as the full
# /mcp URL, so the provider's server_url must match it exactly.
SERVER_URL = MCP_URL

TOKEN_FILE = os.path.join("memory", "indmoney_oauth.json")

CALLBACK_HOST = "localhost"
CALLBACK_PORT = 8765
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

CLIENT_NAME = "Stock Agent (local)"
SCOPE = "portfolio:read"

LOGIN_TIMEOUT_SECONDS = 300


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

class FileTokenStorage:
    """Persist OAuth tokens and client registration to a JSON file."""

    def __init__(self, path=TOKEN_FILE):
        self.path = path

    def _read(self):
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    async def get_tokens(self):
        data = self._read().get("tokens")
        return OAuthToken.model_validate(data) if data else None

    async def set_tokens(self, tokens):
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        data["tokens_saved_at"] = int(time.time())
        self._write(data)

    async def get_client_info(self):
        data = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info):
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)

    # Helpers for status reporting (sync, used outside the auth flow)

    def has_tokens(self):
        return bool(self._read().get("tokens"))

    def token_summary(self):
        data = self._read()
        tokens = data.get("tokens") or {}
        return {
            "has_tokens": bool(tokens),
            "has_refresh_token": bool(tokens.get("refresh_token")),
            "expires_in": tokens.get("expires_in"),
            "scope": tokens.get("scope"),
            "tokens_saved_at": data.get("tokens_saved_at"),
            "client_registered": bool(data.get("client_info")),
        }

    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)


# ---------------------------------------------------------------------------
# Local callback server
# ---------------------------------------------------------------------------

class _CallbackState:
    def __init__(self):
        self.result = None
        self.error = None
        self.event = threading.Event()


def _make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):  # silence default logging
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return

            params = parse_qs(parsed.query)

            if "error" in params:
                state.error = params.get("error_description", params["error"])[0]
                body = f"<h2>INDmoney login failed</h2><p>{state.error}</p>"
            elif "code" in params:
                state.result = AuthorizationCodeResult(
                    code=params["code"][0],
                    state=params.get("state", [None])[0],
                    iss=params.get("iss", [None])[0],
                )
                body = (
                    "<h2>INDmoney connected</h2>"
                    "<p>You can close this tab and return to the terminal.</p>"
                )
            else:
                state.error = "callback received without code"
                body = "<h2>Unexpected callback</h2>"

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            state.event.set()

    return Handler


def _wait_for_callback_blocking(state, timeout):
    server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _make_handler(state))
    server.timeout = 1.0
    deadline = time.time() + timeout
    try:
        while not state.event.is_set() and time.time() < deadline:
            server.handle_request()
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def build_oauth_provider(storage=None, open_browser=True):
    storage = storage or FileTokenStorage()
    state = _CallbackState()

    async def redirect_handler(authorization_url):
        print("\n[indmoney] Opening INDmoney sign-in in your browser.")
        print("[indmoney] If it does not open, visit this URL:\n" + authorization_url + "\n")
        if open_browser:
            webbrowser.open(authorization_url)

    async def callback_handler():
        print(f"[indmoney] Waiting up to {LOGIN_TIMEOUT_SECONDS}s for the login redirect on {REDIRECT_URI} ...")
        await asyncio.to_thread(_wait_for_callback_blocking, state, LOGIN_TIMEOUT_SECONDS)
        if state.error:
            raise RuntimeError(f"INDmoney login failed: {state.error}")
        if state.result is None:
            raise TimeoutError("Timed out waiting for INDmoney login redirect")
        print("[indmoney] Login redirect received; exchanging code for tokens.")
        return state.result

    metadata = OAuthClientMetadata(
        client_name=CLIENT_NAME,
        redirect_uris=[AnyUrl(REDIRECT_URI)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=SCOPE,
        token_endpoint_auth_method="none",
    )

    return OAuthClientProvider(
        server_url=SERVER_URL,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
