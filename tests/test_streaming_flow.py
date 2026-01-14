import json
import sys
import uuid
from pathlib import Path

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.models import OpenAIChatMessage, OpenAIToolCall
from src.openai_transformers import _process_messages, gemini_stream_chunk_to_openai


def test_streaming_reconstruction():
    print("\n--- Test Streaming: Gemini Chunks -> Reconstructed Message -> Gemini Request ---")

    response_id = f"chatcmpl-{uuid.uuid4()}"
    model = "gemini-2.0-flash-thinking-exp"

    # Simulate Gemini streaming chunks
    # Chunk 1: Thinking text
    chunk1 = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "I am thinking about weather.", "thought": True}]},
            },
        ],
    }

    # Chunk 2: More thinking + Signature (might come alone or with text)
    chunk2 = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": " Still thinking.",
                            "thought": True,
                            "thoughtSignature": "streamed_signature_999",
                        },
                    ],
                },
                "finishReason": None,
            },
        ],
    }

    # Chunk 3: Function Call
    chunk3 = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"name": "get_weather", "args": {"location": "Paris"}},
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
    }

    chunks = [chunk1, chunk2, chunk3]

    # 1. Simulate Client Accumulation
    accumulated_message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tool_calls": [],
        "thought_signature": None,
    }

    # Temporary buffer for tool calls construction
    tool_call_buffer = []

    print("Processing Chunks...")
    for i, chunk in enumerate(chunks):
        openai_chunk_dict = gemini_stream_chunk_to_openai(chunk, model, response_id)
        # Convert to Pydantic model to simulate strict client behavior (optional but good practice)
        # openai_chunk = OpenAIChatCompletionStreamResponse(**openai_chunk_dict) # Not strictly necessary for this logic test

        choice = openai_chunk_dict["choices"][0]
        delta = choice["delta"]

        print(f"Chunk {i + 1} Delta: {json.dumps(delta)}")

        # Accumulate
        if delta.get("content"):
            accumulated_message["content"] += delta["content"]

        if delta.get("reasoning_content"):
            accumulated_message["reasoning_content"] += delta["reasoning_content"]

        if delta.get("thought_signature"):
            accumulated_message["thought_signature"] = delta["thought_signature"]
            print(f"  -> Captured Signature: {delta['thought_signature']}")

        if delta.get("tool_calls"):
            # In a real client, we'd need to handle index and merging.
            # Here simplified: just append because Gemini sends full tool calls usually.
            for tc in delta["tool_calls"]:
                tool_call_buffer.append(tc)

    # Finalize message
    accumulated_message["tool_calls"] = tool_call_buffer or None
    if not accumulated_message["content"]:
        accumulated_message["content"] = None  # OpenAI convention

    print("\nReconstructed Message:")
    print(json.dumps(accumulated_message, indent=2))

    # Assertions on reconstruction
    assert accumulated_message["tool_calls"]
    assert accumulated_message["thought_signature"] == "streamed_signature_999"
    assert "I am thinking about weather. Still thinking." in accumulated_message["reasoning_content"]
    assert accumulated_message["tool_calls"][0]["function"]["name"] == "get_weather"

    # 2. Feed back into Transformer (Request construction)
    print("\nConverting back to Gemini format...")

    # Create OpenAIChatMessage object
    msg_obj = OpenAIChatMessage(
        role=accumulated_message["role"],
        content=accumulated_message["content"],
        reasoning_content=accumulated_message["reasoning_content"],
        tool_calls=[OpenAIToolCall(**tc) for tc in accumulated_message["tool_calls"]]
        if accumulated_message["tool_calls"]
        else None,
        thought_signature=accumulated_message["thought_signature"],
    )

    gemini_messages = _process_messages([msg_obj])
    gemini_msg = gemini_messages[0]

    print("\nGemini Message:")
    print(json.dumps(gemini_msg, indent=2))

    # Assertions on Gemini format
    parts = gemini_msg["parts"]

    # Find function call part
    fc_part = next((p for p in parts if "functionCall" in p), None)
    assert fc_part is not None, "FunctionCall part missing"

    # Check signature placement
    if "thoughtSignature" in fc_part:
        print(f"✅ Signature found in functionCall: {fc_part['thoughtSignature']}")
        assert fc_part["thoughtSignature"] == "streamed_signature_999"
    else:
        # Check thought part
        thought_part = next((p for p in parts if "thought" in p), None)
        if thought_part and "thoughtSignature" in thought_part:
            print(f"⚠️ Signature found in thought part: {thought_part['thoughtSignature']}")
            msg = "Signature should be in functionCall part when tool calls exist"
            raise AssertionError(msg)
        msg = "Signature MISSING completely in Gemini message"
        raise AssertionError(msg)

    print("🎉 Streaming Test Passed!")


def test_streaming_reconstruction_no_reasoning_content():
    print("\n--- Test Streaming: Signature ONLY (No Reasoning Text) ---")
    # Case where client might not reconstruct reasoning_content or it was empty

    accumulated_message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": None,  # Missing!
        "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "test_func", "arguments": "{}"}},
        ],
        "thought_signature": "signature_without_text",
    }

    print("Input Message (Simulated):", json.dumps(accumulated_message, indent=2))

    msg_obj = OpenAIChatMessage(
        role=accumulated_message["role"],
        content=accumulated_message["content"],
        reasoning_content=accumulated_message["reasoning_content"],
        tool_calls=[OpenAIToolCall(**tc) for tc in accumulated_message["tool_calls"]],
        thought_signature=accumulated_message["thought_signature"],
    )

    gemini_messages = _process_messages([msg_obj])
    gemini_msg = gemini_messages[0]

    print("\nGemini Message:")
    print(json.dumps(gemini_msg, indent=2))

    parts = gemini_msg["parts"]
    fc_part = next((p for p in parts if "functionCall" in p), None)
    assert fc_part
    assert fc_part["thoughtSignature"] == "signature_without_text"
    print("✅ Signature correctly attached to functionCall even without reasoning text.")


if __name__ == "__main__":
    try:
        test_streaming_reconstruction()
        test_streaming_reconstruction_no_reasoning_content()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
