"""OpenAI-compatible HTTP API and web chat UI.

The wire format is the de-facto standard that llama.cpp's server, vLLM and
most clients speak, so existing tools work unchanged:

    GET  /v1/models
    POST /v1/chat/completions      (streaming via server-sent events)
    POST /v1/completions
    GET  /health, /metrics, /props
    GET  /                         built-in chat page

Only the Python standard library is used.  Requests are served one at a time:
the engine holds a single sequence state, and a follow-up request that extends
the previous conversation reuses it instead of re-reading the prompt.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WEBUI = os.path.join(os.path.dirname(__file__), "webui", "index.html")
MAX_BODY = 64 << 20


def _sampling(req: dict) -> dict:
    out = {k: req.get(k) for k in ("temperature", "top_p", "top_k", "seed", "repetition_penalty")}
    if req.get("repeat_penalty") is not None:
        out["repetition_penalty"] = req["repeat_penalty"]
    return out


def _stops(req: dict) -> list:
    s = req.get("stop")
    return [s] if isinstance(s, str) else list(s or [])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "inferenceSE"
    engine = None
    model_name = "inferencese"
    api_key = None
    busy = threading.Lock()

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ helpers
    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", self.server.cors)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, msg: str, kind="invalid_request_error"):
        self._send(code, {"error": {"message": msg, "type": kind, "code": code}})

    def _authorized(self) -> bool:
        if not self.api_key:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {self.api_key}"

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", self.server.cors)
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj):
        self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
        self.wfile.flush()

    # -------------------------------------------------------------- routes
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.server.cors)
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(WEBUI, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/health":
            return self._send(200, {"status": "ok"})
        if not self._authorized():
            return self._error(401, "invalid API key", "authentication_error")
        if path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{
                "id": self.model_name, "object": "model", "created": int(self.server.started),
                "owned_by": "inferencese"}]})
        if path == "/metrics":
            return self._send(200, self.engine.stats())
        if path == "/props":
            p = self.engine.profile
            return self._send(200, {"profile": p.name, "model": p.model.get("id"),
                                    "description": p.doc.get("description"),
                                    "sampling": p.doc.get("sampling", {}),
                                    "context_length": p.model.get("context_length")})
        self._error(404, f"no route {path}")

    def do_POST(self):
        path = self.path.split("?")[0]
        if not self._authorized():
            return self._error(401, "invalid API key", "authentication_error")
        try:
            req = self._body()
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, f"bad request body: {e}")
        if path == "/v1/chat/completions":
            return self._generate(req, chat=True)
        if path == "/v1/completions":
            return self._generate(req, chat=False)
        if path == "/reset":
            self.engine.reset()
            return self._send(200, {"status": "ok"})
        self._error(404, f"no route {path}")

    # ---------------------------------------------------------- generation
    def _generate(self, req: dict, chat: bool):
        eng = self.engine
        max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 256)
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        stream = bool(req.get("stream"))
        try:
            if chat:
                messages = req.get("messages")
                if not isinstance(messages, list) or not messages:
                    return self._error(400, "'messages' must be a non-empty list")
                targs = {k: req[k] for k in ("tools", "tool_choice", "response_format",
                                             "reasoning_effort") if req.get(k) is not None}
                ids = eng.tokenizer.apply_chat(messages, add_generation_prompt=True, **targs)
            else:
                prompt = req.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = "".join(prompt)
                ids = eng.tokenizer.encode(prompt)
                if getattr(eng.tokenizer, "bos_id", None) is not None and eng.profile.runtime.get("add_bos", True):
                    ids = [eng.tokenizer.bos_id] + ids
        except NotImplementedError as e:
            return self._error(400, str(e))
        except Exception as e:
            return self._error(400, f"could not build the prompt: {type(e).__name__}: {e}")

        if not self.busy.acquire(blocking=bool(req.get("wait", True))):
            return self._error(503, "engine is busy", "server_error")
        try:
            usage = {"prompt_tokens": len(ids), "completion_tokens": 0}
            finish = "length"
            pieces = eng.stream_text(ids, max_new_tokens=max_new, sampling=_sampling(req),
                                     stop=_stops(req), channels=chat, usage=usage)
            if stream:
                self._sse_start()
                if chat:
                    self._sse({"id": rid, "object": obj + ".chunk", "created": created,
                               "model": self.model_name,
                               "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                for channel, piece in pieces:
                    if chat:
                        key = "reasoning_content" if channel == "reasoning" else "content"
                        choice = {"index": 0, "delta": {key: piece}, "finish_reason": None}
                    else:
                        choice = {"index": 0, "text": piece, "finish_reason": None}
                    self._sse({"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                               "model": self.model_name, "choices": [choice]})
                finish = usage.get("finish_reason", finish)
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                tail = {"index": 0, "finish_reason": finish}
                tail["delta" if chat else "text"] = {} if chat else ""
                self._sse({"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                           "model": self.model_name, "choices": [tail],
                           "usage": {k: v for k, v in usage.items() if k.endswith("_tokens")},
                           "timings": eng.stats()["timing"]})
                self._sse(b"[DONE]")
                return
            text = {"content": "", "reasoning": "", "tool_calls": ""}
            for channel, piece in pieces:
                text[channel if channel in text else "content"] += piece
            finish = usage.get("finish_reason", finish)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            usage = {k: v for k, v in usage.items() if k.endswith("_tokens")}
            if chat:
                msg = {"role": "assistant", "content": text["content"]}
                if text["reasoning"]:
                    msg["reasoning_content"] = text["reasoning"]
                choice = {"index": 0, "message": msg, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text["content"], "finish_reason": finish}
            self._send(200, {"id": rid, "object": obj, "created": created, "model": self.model_name,
                             "choices": [choice], "usage": usage, "timings": eng.stats()["timing"]})
        except (BrokenPipeError, ConnectionResetError):
            pass                                    # client went away mid-stream
        except Exception as e:
            try:
                self._error(500, f"{type(e).__name__}: {e}", "server_error")
            except Exception:
                pass
        finally:
            self.busy.release()


def serve(engine, host: str = "127.0.0.1", port: int = 8080, api_key: str | None = None,
          cors: str = "*", verbose: bool = False, model_name: str | None = None):
    Handler.engine = engine
    Handler.api_key = api_key
    Handler.model_name = model_name or engine.profile.model.get("id") or engine.profile.name
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.verbose, httpd.cors, httpd.started = verbose, cors, time.time()
    return httpd
