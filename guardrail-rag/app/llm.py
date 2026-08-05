from __future__ import annotations

import json
import os
from typing import Any
from app.logging_config import get_logger

logger = get_logger(__name__)


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


def _build_google_payload(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [
                    {"text": f"System: {system_prompt}\n\nUser: {user_prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str | None:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    logger.info("call_llm: provider=%s temperature=%s", provider, temperature)

    if provider in {"google", "gemini"}:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            return None
        model = os.getenv("GOOGLE_MODEL", os.getenv("LLM_MODEL", "gemini-1.5-flash"))
        payload = _build_google_payload(system_prompt, user_prompt, temperature)
        headers = {"Content-Type": "application/json"}
        base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models")
        url = f"{base_url.rstrip('/')}/{model}:generateContent?key={api_key}"
    else:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            return None

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

    logger.debug("call_llm: calling google model url=%s payload_keys=%s", url, list(payload.keys()))
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            if provider in {"google", "gemini"}:
                candidates = body.get("candidates", [])
                if not candidates:
                    return None
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return None
                return parts[0].get("text", "")
            result = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info("call_llm: received response (len=%d)", len(result) if result else 0)
            return result
    except Exception:
        logger.exception("call_llm: error calling LLM provider=%s", provider)
        return None
