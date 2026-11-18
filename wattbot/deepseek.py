"""Use only DeepSeek V4.1 Flash 1017 for generative calls."""

import json
import os
import time
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash-1017"
CAMPUS_MODEL = "DeepSeek-V4.1-Flash"
CAMPUS_BASE = "https://aigc-api.hkust-gz.edu.cn/api/v2"


def setting(name):
    value = os.environ.get(name)
    if value:
        return value
    dotenv = Path(__file__).resolve().parents[1] / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, _, value = line.strip().removeprefix("export ").partition("=")
            if key.strip() == name:
                return value.strip().strip(chr(34) + chr(39))
    return None


class DeepSeek:
    def __init__(self, backend, *, timeout=90, transport_attempts=3):
        self.backend = backend
        self.timeout = timeout
        self.transport_attempts = transport_attempts
        if backend == "hkustgz":
            config_root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            config = config_root / "opencode" / "opencode.json"
            key = setting("HKUSTGZ_AIGC_API_KEY")
            if not key:
                options = json.loads(config.read_text(encoding="utf-8"))[
                    "provider"]["hkustgz-aigc"]["options"]
                if options["baseURL"].rstrip("/") != CAMPUS_BASE:
                    raise ValueError("Unexpected HKUSTGZ endpoint in the OpenCode configuration")
                key = options["apiKey"]
            self.key = key
            self.url = CAMPUS_BASE + "/chat/completions"
            self.model = CAMPUS_MODEL
            self.accepted = {CAMPUS_MODEL, "DeepSeek-V4.1-Flash-1017"}
        elif backend == "openrouter":
            self.key = setting("OPENROUTER_API_KEY")
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = OPENROUTER_MODEL
            self.accepted = {self.model, "deepseek/deepseek-v4.1-flash-20261017"}
        else:
            raise ValueError(f"Unknown LLM backend {backend}")
        if not self.key or self.key.startswith("{"):
            raise ValueError(f"Configure a usable API key for {backend}")

    def complete(self, stage, system, data, emit):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]
        payload = {
            "model": self.model, "messages": messages, "temperature": 0,
            "max_tokens": 2400, "response_format": {"type": "json_object"},
        }
        if self.backend == "openrouter":
            payload["reasoning"] = {"enabled": False}
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        request = Request(self.url, data=json.dumps(payload).encode(), method="POST",
                          headers={"Authorization": f"Bearer {self.key}",
                                   "Content-Type": "application/json"})
        for attempt in range(1, self.transport_attempts + 1):
            emit("llm_request", stage=stage, attempt=attempt, model=self.model, backend=self.backend, messages=messages)
            started = time.monotonic()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                break
            except HTTPError as error:
                status = error.code
                emit("llm_http_error", stage=stage, attempt=attempt, status=status)
                if status not in {408, 429, 500, 502, 503, 504} or attempt == self.transport_attempts:
                    raise RuntimeError(f"LLM endpoint returned HTTP {status} in {stage}") from error
                time.sleep(attempt * 2)
            except (URLError, TimeoutError) as error:
                emit("llm_network_error", stage=stage, attempt=attempt, detail=str(error))
                if attempt == self.transport_attempts:
                    raise RuntimeError(f"LLM network error in {stage}: {error}") from error
                time.sleep(attempt * 2)
        if not body.get("choices"):
            emit("llm_response_error", stage=stage, response_id=body.get("id"))
            raise RuntimeError(f"LLM endpoint returned no choice in {stage}")
        choice = body["choices"][0]
        content = choice.get("message", {}).get("content")
        emit("llm_response", stage=stage, model=body.get("model"), provider=body.get("provider"),
             response_id=body.get("id"), system_fingerprint=body.get("system_fingerprint"), usage=body.get("usage"), content=content,
             finish_reason=choice.get("finish_reason"), elapsed_ms=round(
                 (time.monotonic() - started) * 1000))
        if body.get("model") not in self.accepted:
            raise RuntimeError(f"Unexpected served model in {stage}: {body.get('model')!r}")
        if choice.get("finish_reason") not in {None, "stop"}:
            raise ValueError(f"Unfinished completion in {stage}")
        result = json.loads(content or "", parse_float=Decimal)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object in {stage}")
        return result
