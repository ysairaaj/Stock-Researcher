"""
Thin synchronous client for the INDmoney MCP server.

One session per call keeps the code simple; the holdings aggregator caches
results so the number of calls stays small.
"""
import asyncio
import json

import httpx2

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from brokers.indmoney.auth import MCP_URL, FileTokenStorage, build_oauth_provider


HTTP_TIMEOUT = httpx2.Timeout(30.0, read=120.0)


def _unwrap(result):
    """
    Normalise a CallToolResult into Python data.

    INDmoney wraps payloads as {"result": "<json string>"} inside a text
    content block; unwrap that, otherwise return parsed JSON or raw text.
    """
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured:
        data = structured
    else:
        texts = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            if text is not None:
                texts.append(text)
        raw = "\n".join(texts).strip()
        try:
            data = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return raw

    # Unwrap {"result": "..."} envelopes, possibly nested JSON strings
    for _ in range(3):
        if isinstance(data, dict) and set(data.keys()) == {"result"}:
            inner = data["result"]
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    return inner
            data = inner
        else:
            break

    return data


async def _with_session(fn, open_browser=True):
    oauth = build_oauth_provider(open_browser=open_browser)
    async with httpx2.AsyncClient(auth=oauth, timeout=HTTP_TIMEOUT) as http:
        async with streamable_http_client(MCP_URL, http_client=http) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (unlikely for this CLI agent): run in a thread.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def is_connected():
    return FileTokenStorage().has_tokens()


def token_status():
    return FileTokenStorage().token_summary()


def connect():
    """Run (or refresh) the OAuth flow by making a trivial authenticated call."""
    async def fn(session):
        tools = await session.list_tools()
        return [t.name for t in tools.tools]

    tool_names = _run(_with_session(fn, open_browser=True))
    return {"connected": True, "tools": tool_names, **token_status()}


def disconnect():
    FileTokenStorage().clear()
    return {"connected": False}


def list_tools():
    """Return [{name, description, inputSchema}] for every server tool."""
    async def fn(session):
        res = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None),
            }
            for t in res.tools
        ]

    return _run(_with_session(fn))


def call_tool(name, arguments=None):
    async def fn(session):
        res = await session.call_tool(name, arguments or {})
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            raise RuntimeError(f"INDmoney tool {name} returned error: {_unwrap(res)}")
        return _unwrap(res)

    return _run(_with_session(fn))
