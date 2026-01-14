import json
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.models import OpenAIChatMessage, OpenAIToolCall
from src.openai_transformers import _process_messages, gemini_response_to_openai


def test_gemini_to_openai_with_signature():
    print("\n--- Test 1: Gemini Response -> OpenAI Format (Extract Signature) ---")

    # Mock a Gemini response with Thought (Signed) + Function Call
    # This structure mimics what Gemini 2.0 Flash Thinking returns
    gemini_response = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "I will check the weather for London.",
                            "thought": True,
                            "thoughtSignature": "super_secret_signature_123",
                        },
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "London, UK"},
                            },
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
    }

    print("Input Gemini Response:", json.dumps(gemini_response, indent=2))

    openai_response = gemini_response_to_openai(gemini_response, model="gemini-2.0-flash-thinking-exp")
    choice = openai_response["choices"][0]
    message = choice["message"]

    print("\nOutput OpenAI Message:")
    print(json.dumps(message, indent=2))

    # Assertions
    assert message["role"] == "assistant"
    assert message["tool_calls"] is not None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert message["thought_signature"] == "super_secret_signature_123"
    assert "GEMINI_SIG" not in message["reasoning_content"]

    print("✅ Test 1 Passed: Signature successfully stored in thought_signature field.")
    return message


def test_openai_to_gemini_with_signature(assistant_message_data):
    print("\n--- Test 2: OpenAI Request -> Gemini Format (Restore Signature & Format FunctionResponse) ---")

    # Construct the history as if it came back from the client
    # 1. User: "Weather in London?"
    # 2. Assistant: (The message from Test 1, containing the thought_signature field)
    # 3. Tool: Result of the function call

    messages = [
        OpenAIChatMessage(role="user", content="What is the weather in London?"),
        OpenAIChatMessage(
            role="assistant",
            # The client sends back exactly what we sent them
            reasoning_content=assistant_message_data["reasoning_content"],
            # Explicitly pass the signature field
            thought_signature=assistant_message_data["thought_signature"],
            tool_calls=[OpenAIToolCall(**tc) for tc in assistant_message_data["tool_calls"]],
        ),
        OpenAIChatMessage(
            role="tool",
            tool_call_id=assistant_message_data["tool_calls"][0]["id"],
            content=json.dumps({"temperature": "15C", "condition": "Cloudy"}),
        ),
    ]

    print("Input OpenAI Messages (History + Tool Output):")
    for m in messages:
        print(f"[{m.role}] {m.content or '...'} (Reasoning: {str(m.reasoning_content)[:30]}..., Sig: {m.thought_signature})")

    # Transform to Gemini format
    gemini_contents = _process_messages(messages)

    print("\nOutput Gemini Contents:")
    print(json.dumps(gemini_contents, indent=2))

    # Verify Step 2 (Assistant Message) - Check Signature Placement
    assistant_part = gemini_contents[1]
    assert assistant_part["role"] == "model"

    # According to new logic, signature should be on the FunctionCall part if tool calls exist
    parts = assistant_part["parts"]

    # Find the function call part
    fc_part = next((p for p in parts if "functionCall" in p), None)
    thought_part = next((p for p in parts if "thought" in p), None)  # Should be converted to thought: True

    assert fc_part is not None, "Missing functionCall part"
    assert thought_part is not None, "Missing thought part"

    # CRITICAL CHECK: Signature must be in the functionCall part
    print("Checking signature location...")
    if "thoughtSignature" in fc_part:
        print(f"Found signature in functionCall: {fc_part['thoughtSignature']}")
        assert fc_part["thoughtSignature"] == "super_secret_signature_123"
    else:
        print("Signature NOT found in functionCall part!")
        print("FC Part keys:", fc_part.keys())
        msg = "Signature should be attached to the functionCall part when tool calls are present."
        raise AssertionError(msg)

    # Verify Step 3 (Tool Response) - Check functionResponse structure
    tool_part = gemini_contents[2]
    assert tool_part["role"] == "user", "Tool response role must be 'user'"
    assert "functionResponse" in tool_part["parts"][0], "Missing functionResponse"

    fr = tool_part["parts"][0]["functionResponse"]
    assert fr["name"] == "get_weather", "Correct function name resolved"
    assert fr["response"]["temperature"] == "15C", "Response content preserved"

    print("✅ Test 2 Passed: Signature restored to correct part and functionResponse formatted correctly.")


if __name__ == "__main__":
    try:
        assistant_msg = test_gemini_to_openai_with_signature()
        test_openai_to_gemini_with_signature(assistant_msg)
        print("\n🎉 ALL TESTS PASSED")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
