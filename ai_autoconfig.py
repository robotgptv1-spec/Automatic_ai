"""
ai_autoconfig.py
-----------------
Lets the user paste in their own OpenAI / Anthropic / Gemini API key plus a
plain-English prompt ("predict whether a customer will churn"), and the LLM
picks the target column, the feature columns, and the task type from the
dataset's schema — instead of the user picking them by hand in Step 2.

The key is never stored server-side: it's forwarded straight through on the
single request and discarded when the request finishes (SESSIONS only keeps
the dataset/model, never the key).

Each provider function returns raw text; `suggest_config` is the only
function app.py calls — it prompts the model, strips markdown fences if the
model wrapped the JSON in them, and parses it.
"""

import json
import re
import requests

# Bump these if you want a different default model per provider.
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
GEMINI_MODEL = "gemini-2.0-flash"

REQUEST_TIMEOUT = 30


class AIConfigError(Exception):
    """Raised for any provider/network/parsing failure — caught in app.py
    and returned to the frontend as a normal error message."""


def _schema_text(columns_info):
    lines = []
    for c in columns_info:
        samples = ", ".join(str(s) for s in c["sample_values"][:4])
        lines.append(
            f"- {c['name']} (dtype={c['dtype']}, unique_values={c['nunique']}, "
            f"nulls={c['null_count']}, examples=[{samples}])"
        )
    return "\n".join(lines)


def _build_prompt(user_prompt, columns_info):
    schema = _schema_text(columns_info)
    column_names = [c["name"] for c in columns_info]
    system = (
        "You are configuring a small AutoML tool. Given a dataset's column "
        "schema and a user's goal in plain English, decide: the single best "
        "target column to predict, the feature columns to use (a subset of "
        "the remaining columns, drop obvious ID/leak columns), and whether "
        "the task is classification or regression. "
        "Respond with ONLY a raw JSON object, no markdown fences, no prose, "
        "matching exactly this shape:\n"
        '{"target_column": "<one column name>", '
        '"feature_columns": ["<column name>", "..."], '
        '"task_type": "classification" | "regression", '
        '"reasoning": "<one short sentence>"}\n'
        f"Valid column names are exactly: {column_names}"
    )
    user = f"Dataset columns:\n{schema}\n\nUser's goal: {user_prompt}"
    return system, user


def _extract_json(text):
    text = text.strip()
    # strip ```json ... ``` or ``` ... ``` fences if the model added them anyway
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: grab the first {...} blob in the text
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise AIConfigError("The model didn't return valid JSON. Try rephrasing the prompt.")


def _call_openai(api_key, system, user):
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise AIConfigError(f"Could not reach OpenAI: {e}")
    if resp.status_code != 200:
        raise AIConfigError(f"OpenAI error ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(api_key, system, user):
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 500,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise AIConfigError(f"Could not reach Anthropic: {e}")
    if resp.status_code != 200:
        raise AIConfigError(f"Anthropic error ({resp.status_code}): {resp.text[:300]}")
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _call_gemini(api_key, system, user):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    try:
        resp = requests.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        raise AIConfigError(f"Could not reach Gemini: {e}")
    if resp.status_code != 200:
        raise AIConfigError(f"Gemini error ({resp.status_code}): {resp.text[:300]}")
    candidates = resp.json().get("candidates", [])
    if not candidates:
        raise AIConfigError("Gemini returned no candidates — check the API key and quota.")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


_PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
}


def suggest_config(provider, api_key, user_prompt, columns_info):
    """Returns {"target_column", "feature_columns", "task_type", "reasoning"}."""
    if provider not in _PROVIDERS:
        raise AIConfigError(f"Unknown provider: {provider}")
    system, user = _build_prompt(user_prompt, columns_info)
    raw_text = _PROVIDERS[provider](api_key, system, user)
    if not raw_text or not raw_text.strip():
        raise AIConfigError("The model returned an empty response.")
    parsed = _extract_json(raw_text)
    if "target_column" not in parsed:
        raise AIConfigError("The model's response was missing a target_column.")
    return parsed
