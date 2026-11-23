"""Claude backend for the LLM steps (weight edits in llm_edit.py, law reading in story3/reader.py), through the
official Anthropic Python SDK. Used whenever the model id starts with "claude-" (e.g. claude-opus-5-5,
claude-sonnet-5-5); every other id keeps going to CBorg.

Key: ANTHROPIC_API_KEY in the environment (the job scripts `source ~/.anthropic_key` if that file exists).
Thinking: adaptive (always on for Opus 5.5 / Sonnet 5.5); depth set by CLAUDE_EFFORT (default high).
Refusals: server-side fallback (fallbacks="default", beta) unless CLAUDE_FALLBACKS=0. The model that actually answered
is recorded in usage["served_by"] (and a "fallback" flag), so fallback-answered calls can be flagged in the analysis.
429 / 5xx / connection errors are retried by the SDK (max_retries below)."""
import os

import anthropic

_client = None


def _get():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=10, timeout=900.0)
    return _client


def call(model_id, messages, max_tokens=32000):
    """messages: chat list as used by the CBorg path (system messages first, then user/assistant turns).
    Returns (text, usage dict) like llm_edit.call / reader.ask."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    kw = dict(model=model_id, max_tokens=max_tokens, messages=turns,
              output_config={"effort": os.environ.get("CLAUDE_EFFORT", "high")})
    if system:
        kw["system"] = system
    fallbacks = os.environ.get("CLAUDE_FALLBACKS", "1") != "0"
    if fallbacks:
        kw.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        stream = _get().beta.messages.stream(**kw)
    else:
        stream = _get().messages.stream(**kw)
    with stream as s:   # streaming: long thinking at high effort stays clear of HTTP timeouts
        msg = s.get_final_message()
    fell_back = any(getattr(it, "type", "") == "fallback_message" for it in (getattr(msg.usage, "iterations", None) or []))
    if msg.stop_reason == "refusal":
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        raise RuntimeError(f"Claude declined the request (refusal, category {cat})")
    text = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    usage = {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens, "served_by": msg.model,
             "fallback": fell_back, "stop_reason": msg.stop_reason, "request_id": msg._request_id,
             "effort": kw["output_config"]["effort"]}
    return text, usage
