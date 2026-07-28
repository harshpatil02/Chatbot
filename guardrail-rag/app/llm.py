from __future__ import annotations

import json
import os
from typing import Any


def _build_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _build_openai_payload(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
    return {
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }


def _build_azure_payload(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str | None:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        return None

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "azure":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            return None
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            return None
        payload = _build_azure_payload(system_prompt, user_prompt, temperature)
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-01"
    else:
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        payload = _build_openai_payload(system_prompt, user_prompt, temperature)
        headers = _build_headers(api_key)
        url = endpoint

    import urllib.request

    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return None
