"""
OpenAI Format Transformers - Handles conversion between OpenAI and Gemini API formats.
This module contains all the logic for transforming requests and responses between the two formats.
"""

import json
import re
import time
import uuid
from typing import Any

from .config import (
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    get_thinking_budget,
    is_maxthinking_model,
    is_nothinking_model,
    is_search_model,
    should_include_thoughts,
)
from .models import OpenAIChatCompletionRequest, OpenAIChatMessage


def openai_request_to_gemini(openai_request: OpenAIChatCompletionRequest) -> dict[str, Any]:
    """
    Transform an OpenAI chat completion request to Gemini format.

    Args:
        openai_request: OpenAI format request

    Returns:
        Dictionary in Gemini API format
    """
    contents = _process_messages(openai_request.messages)
    generation_config = _map_generation_config(openai_request)

    # Build the base request payload
    request_payload = {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": get_base_model_name(openai_request.model),
    }

    # Handle tools/function calling
    if openai_request.tools:
        request_payload["tools"] = _map_openai_tools_to_gemini(openai_request.tools)

    if openai_request.tool_choice:
        request_payload["toolConfig"] = _map_tool_config(openai_request.tool_choice)

    # Add Google Search grounding for search models
    if is_search_model(openai_request.model):
        if "tools" not in request_payload:
            request_payload["tools"] = []
        # Check if googleSearch is already added (unlikely but safe to check)
        if not any("googleSearch" in tool for tool in request_payload["tools"]):
            request_payload["tools"].append({"googleSearch": {}})

    # Configure thinking budget
    _configure_thinking(request_payload, openai_request.model, openai_request)

    return request_payload


def _process_messages(messages: list[OpenAIChatMessage]) -> list[dict[str, Any]]:
    """Process all messages, handling tool call name resolution."""
    contents = []
    # Map tool_call_id to function name from assistant messages
    tool_id_to_name = {}

    for message in messages:
        # 1. Capture tool definitions from assistant messages
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_id_to_name[tool_call.id] = tool_call.function["name"]

        # 2. Process the message
        gemini_msg = _process_single_message(message)

        # 3. Fix up tool responses (add function name)
        if message.role == "tool" and message.tool_call_id:
            # It's a tool response. We need to wrap the content in functionResponse
            # and hopefully we found the name.
            fn_name = tool_id_to_name.get(message.tool_call_id, "unknown_function")

            # The content in OpenAI is usually a string (JSON).
            # Gemini expects 'response' to be a dict (struct).
            response_content = _parse_json_safe(message.content) if message.content else {}
            if not isinstance(response_content, (dict, list)):
                # If primitive, wrap it
                response_content = {"result": response_content}

            gemini_msg["parts"] = [
                {
                    "functionResponse": {
                        "name": fn_name,
                        "response": response_content,
                    },
                },
            ]
            # Ensure role is 'user' (Gemini spec for functionResponse)
            gemini_msg["role"] = "user"

        contents.append(gemini_msg)

    return contents


def _process_single_message(message: OpenAIChatMessage) -> dict[str, Any]:
    """Process a single message into Gemini format."""
    role = message.role
    if role == "assistant":
        role = "model"
    elif role == "system":
        role = "user"  # Gemini treats system messages as user messages
    elif role == "tool":
        role = "function"  # Initial mapping, refined in _process_messages

    parts = []
    thought_signature = None
    thought_text = None

    # 1. Check for explicit signature (independent of reasoning text)
    if message.thought_signature:
        thought_signature = message.thought_signature

    # 2. Handle thoughts (reasoning_content) extraction
    if message.reasoning_content:
        thought_text = message.reasoning_content
        # Check for embedded signature (fallback if explicit not present)
        if not thought_signature and thought_text.startswith("GEMINI_SIG:"):
            try:
                # Format: GEMINI_SIG:<signature>:<actual_text>
                _, signature, actual_text = thought_text.split(":", 2)
                thought_text = actual_text
                thought_signature = signature
            except ValueError:
                # If splitting fails, treat strictly as text
                pass

    thought_part = None
    if thought_text is not None:
        thought_part = {
            "text": thought_text,
            "thought": True,
        }
        parts.append(thought_part)

    # 3. Handle content
    if message.content:
        if isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.extend(_extract_text_parts(part.get("text", "") or ""))
                    elif part.get("type") == "image_url":
                        parts.append(_process_image_url(part.get("image_url", {}).get("url")))
        else:
            # String content
            parts.extend(_extract_text_parts(message.content or ""))

    # 4. Handle tool calls in assistant messages
    tool_call_parts = []
    if message.tool_calls:
        tool_call_parts.extend(
            {
                "functionCall": {
                    "name": tool_call.function["name"],
                    "args": tool_call.function["arguments"]
                    if isinstance(tool_call.function["arguments"], dict)
                    else _parse_json_safe(tool_call.function["arguments"]),
                },
            }
            for tool_call in message.tool_calls
        )
        parts.extend(tool_call_parts)

    # 5. Attach thoughtSignature to the correct part
    if thought_signature:
        if tool_call_parts:
            # If tool calls exist, signature MUST be on the first functionCall part
            tool_call_parts[0]["thoughtSignature"] = thought_signature
        elif thought_part:
            # Otherwise, attach to the thought part
            thought_part["thoughtSignature"] = thought_signature
        else:
            # If no tool calls and no thought part exists, create a dummy thought part
            # This handles cases where client reconstructs message with signature but no thought text
            dummy_thought_part = {
                "text": "",  # Empty text
                "thought": True,
                "thoughtSignature": thought_signature,
            }
            # Insert at the beginning
            parts.insert(0, dummy_thought_part)

    # Handle tool responses (role='tool' in OpenAI)
    if role == "function" and message.tool_call_id:
        pass  # Handled in _process_messages

    return {"role": role, "parts": parts}


def _extract_text_parts(text: str) -> list[dict[str, Any]]:
    """Extracts text and inline images from markdown text."""
    parts = []
    # Convert Markdown images: ![alt](data:<mimeType>;base64,<data>)
    # Fixed regex to avoid nested set warning
    pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    last_idx = 0
    for m in pattern.finditer(text):
        url = m.group(1).strip().strip('"').strip("'")
        if m.start() > last_idx:
            before = text[last_idx : m.start()]
            if before:
                parts.append({"text": before})

        if url.startswith("data:"):
            try:
                header, base64_data = url.split(",", 1)
                mime_type = ""
                if ":" in header:
                    mime_type = header.split(":", 1)[1].split(";", 1)[0] or ""
                if mime_type.startswith("image/"):
                    parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_data,
                        },
                    })
                else:
                    parts.append({"text": text[m.start() : m.end()]})
            except Exception:
                parts.append({"text": text[m.start() : m.end()]})
        else:
            parts.append({"text": text[m.start() : m.end()]})
        last_idx = m.end()

    if last_idx < len(text):
        tail = text[last_idx:]
        if tail:
            parts.append({"text": tail})

    return parts


def _process_image_url(url: str | None) -> dict[str, Any]:
    """Process an image URL (data URI) into a Gemini part."""
    if not url:
        return {}
    try:
        mime_type, base64_data = url.split(";")
        _, mime_type = mime_type.split(":")
        _, base64_data = base64_data.split(",")
    except ValueError:
        return {}

    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64_data,
        },
    }


def _map_generation_config(openai_request: OpenAIChatCompletionRequest) -> dict[str, Any]:
    """Map OpenAI generation parameters to Gemini format."""
    config = {}
    if openai_request.temperature is not None:
        config["temperature"] = openai_request.temperature
    if openai_request.top_p is not None:
        config["topP"] = openai_request.top_p
    if openai_request.max_tokens is not None:
        config["maxOutputTokens"] = openai_request.max_tokens
    if openai_request.stop is not None:
        if isinstance(openai_request.stop, str):
            config["stopSequences"] = [openai_request.stop]
        elif isinstance(openai_request.stop, list):
            config["stopSequences"] = openai_request.stop
    if openai_request.frequency_penalty is not None:
        config["frequencyPenalty"] = openai_request.frequency_penalty
    if openai_request.presence_penalty is not None:
        config["presencePenalty"] = openai_request.presence_penalty
    if openai_request.n is not None:
        config["candidateCount"] = openai_request.n
    if openai_request.seed is not None:
        config["seed"] = openai_request.seed
    if openai_request.response_format is not None and openai_request.response_format.get("type") == "json_object":
        config["responseMimeType"] = "application/json"
    return config


def _configure_thinking(request_payload: dict, model: str, openai_request: OpenAIChatCompletionRequest):
    """Configures thinking budget if applicable."""
    if "gemini-2.5-flash-image" in model:
        return

    thinking_budget = None
    if is_nothinking_model(model) or is_maxthinking_model(model):
        thinking_budget = get_thinking_budget(model)
    else:
        reasoning_effort = getattr(openai_request, "reasoning_effort", None)
        if reasoning_effort:
            base_model = get_base_model_name(model)
            if reasoning_effort == "minimal":
                if "gemini-2.5-flash" in base_model:
                    thinking_budget = 0
                elif "gemini-2.5-pro" in base_model or "gemini-3-pro" in base_model:
                    thinking_budget = 128
            elif reasoning_effort == "low":
                thinking_budget = 1000
            elif reasoning_effort == "medium":
                thinking_budget = -1
            elif reasoning_effort == "high":
                if "gemini-2.5-flash" in base_model:
                    thinking_budget = 24576
                elif "gemini-2.5-pro" in base_model:
                    thinking_budget = 32768
                elif "gemini-3-pro" in base_model:
                    thinking_budget = 45000
        else:
            thinking_budget = get_thinking_budget(model)

    if thinking_budget is not None:
        if "generationConfig" not in request_payload:
            request_payload["generationConfig"] = {}
        request_payload["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": thinking_budget,
            "includeThoughts": should_include_thoughts(model),
        }


def _map_openai_tools_to_gemini(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tools definition to Gemini format."""
    gemini_tools = []
    for tool in openai_tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            gemini_tool = {
                "functionDeclarations": [
                    {
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "parameters": func.get("parameters"),
                    },
                ],
            }
            gemini_tools.append(gemini_tool)
    return gemini_tools


def _map_tool_config(tool_choice: str | dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI tool_choice to Gemini toolConfig."""
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        # Forced function call
        fn_name = tool_choice.get("function", {}).get("name")
        return {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [fn_name],
            },
        }
    return {}


def _parse_json_safe(content: Any) -> Any:
    """Safely parse JSON string, returning original content if failure."""
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except Exception:
        return content


def gemini_response_to_openai(gemini_response: dict[str, Any], model: str) -> dict[str, Any]:
    """
    Transform a Gemini API response to OpenAI chat completion format.
    """
    choices = []

    for candidate in gemini_response.get("candidates", []):
        choice = _process_candidate(candidate)
        choices.append(choice)

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def _process_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Process a single Gemini candidate into an OpenAI choice."""
    role = candidate.get("content", {}).get("role", "assistant")
    if role == "model":
        role = "assistant"

    parts = candidate.get("content", {}).get("parts", [])
    content_parts = []
    raw_reasoning_text = ""
    active_signature = None
    tool_calls = []

    for part in parts:
        # Check for signature in ANY part
        sig = part.get("thoughtSignature") or part.get("thought_signature")
        if sig:
            active_signature = sig

        # 1. Text / Thinking
        if part.get("text") is not None:
            if part.get("thought", False):
                raw_reasoning_text += part.get("text", "")
            else:
                content_parts.append(part.get("text", ""))

        # 2. Inline Images
        elif part.get("inlineData"):
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or "image/png"
                if isinstance(mime, str) and mime.startswith("image/"):
                    data_b64 = inline.get("data")
                    content_parts.append(f"![image](data:{mime};base64,{data_b64})")

        # 3. Function Calls
        elif part.get("functionCall"):
            fc = part.get("functionCall")
            tool_calls.append({
                "id": "call_" + str(uuid.uuid4())[:8],  # Gemini doesn't provide ID, so generate one
                "type": "function",
                "function": {
                    "name": fc.get("name"),
                    "arguments": import_json_dumps(fc.get("args", {})),  # OpenAI expects arguments as JSON string
                },
            })

    content = "\n\n".join([p for p in content_parts if p]) if content_parts else None
    # Construct reasoning content (clean, without signature prefix)
    reasoning_content = raw_reasoning_text
    message = {
        "role": role,
        "content": content,
    }
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    # Store signature in a dedicated field
    if active_signature:
        message["thought_signature"] = active_signature
    if tool_calls:
        message["tool_calls"] = tool_calls
        if content is None:
            message["content"] = None  # OpenAI allows null content if tool_calls exist
    return {
        "index": candidate.get("index", 0),
        "message": message,
        "finish_reason": _map_finish_reason(candidate.get("finishReason")),
    }


def import_json_dumps(obj: Any) -> str:
    return json.dumps(obj)


def gemini_stream_chunk_to_openai(gemini_chunk: dict[str, Any], model: str, response_id: str) -> dict[str, Any]:
    """
    Transform a Gemini streaming response chunk to OpenAI streaming format.
    """
    choices = []

    for candidate in gemini_chunk.get("candidates", []):
        role = candidate.get("content", {}).get("role", "assistant")
        if role == "model":
            role = "assistant"

        parts = candidate.get("content", {}).get("parts", [])
        content_parts = []
        raw_reasoning_text = ""
        active_signature = None
        tool_calls = []

        for part in parts:
            # Check for signature in ANY part
            sig = part.get("thoughtSignature") or part.get("thought_signature")
            if sig:
                active_signature = sig

            if part.get("text") is not None:
                if part.get("thought", False):
                    raw_reasoning_text += part.get("text", "")
                else:
                    content_parts.append(part.get("text", ""))
            elif part.get("functionCall"):
                # Handle streaming function call
                # Note: Gemini often sends full function call in one chunk in streaming too.
                # If it splits, we might need state, but usually it's full.
                fc = part.get("functionCall")
                tool_calls.append({
                    "id": "call_" + str(uuid.uuid4())[:8],
                    "type": "function",
                    "function": {
                        "name": fc.get("name"),
                        "arguments": import_json_dumps(fc.get("args", {})),
                    },
                })

        content = "\n\n".join([p for p in content_parts if p])

        reasoning_content = raw_reasoning_text

        delta = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        if tool_calls:
            delta["tool_calls"] = tool_calls
        if active_signature:
            delta["thought_signature"] = active_signature

        choices.append({
            "index": candidate.get("index", 0),
            "delta": delta,
            "finish_reason": _map_finish_reason(candidate.get("finishReason")),
        })

    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def _map_finish_reason(gemini_reason: str | None) -> str | None:
    """Map Gemini finish reasons to OpenAI finish reasons."""
    if not gemini_reason:
        return None
    if gemini_reason == "STOP":
        return "stop"
    if gemini_reason == "MAX_TOKENS":
        return "length"
    if gemini_reason in {"SAFETY", "RECITATION"}:
        return "content_filter"
    return None
