import json
import threading
import urllib.error
import urllib.request

import pytest

from inferencese.runtime.engine import Engine
from inferencese.server.api import serve


@pytest.fixture
def server(tiny_llama):
    eng = Engine("generic-llama", tiny_llama, log=None, overrides={
        "runtime": {"backend": "numpy"},
        "memory": {"tiers": {"ram": {"capacity": "32MiB"}, "ssd": {"direct_io": False}}, "warmup": []}})
    httpd = serve(eng, "127.0.0.1", 0, api_key="secret")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    eng.close()


def call(url, body=None, key="secret"):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()


def test_models_health_and_ui(server):
    assert json.loads(call(server + "/health", key=None)[1]) == {"status": "ok"}
    models = json.loads(call(server + "/v1/models")[1])
    assert models["data"][0]["object"] == "model"
    status, page = call(server + "/", key=None)
    assert status == 200 and b"inferenceSE" in page
    metrics = json.loads(call(server + "/metrics")[1])
    assert metrics["memory"]["blocks"] > 0


def test_chat_completion_and_streaming(server):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "temperature": 0}
    out = json.loads(call(server + "/v1/chat/completions", body)[1])
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["role"] == "assistant"
    assert out["usage"]["completion_tokens"] == 5 and out["choices"][0]["finish_reason"] == "length"
    _, raw = call(server + "/v1/chat/completions", dict(body, stream=True))
    events = [l[6:] for l in raw.decode().split("\n") if l.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == out["choices"][0]["message"]["content"]      # same prompt, greedy


def test_text_completion(server):
    out = json.loads(call(server + "/v1/completions", {"prompt": "abc", "max_tokens": 3, "temperature": 0})[1])
    assert out["object"] == "text_completion" and out["usage"]["prompt_tokens"] == 3


def test_errors(server):
    for url, body, key, code in [("/v1/models", None, "wrong", 401),
                                 ("/v1/chat/completions", {"messages": []}, "secret", 400),
                                 ("/v1/nothing", {}, "secret", 404)]:
        with pytest.raises(urllib.error.HTTPError) as e:
            call(server + url, body, key)
        assert e.value.code == code
        assert "error" in json.loads(e.value.read())
