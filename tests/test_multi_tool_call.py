import json
import sys
import uuid
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.models import OpenAIChatMessage, OpenAIToolCall
from src.openai_transformers import _process_messages


def test_multi_tool_call_grouping():
    print("\n--- Test: Grouping Multiple Tool Responses ---")

    # define tool call IDs
    call_id_1 = f"call_{uuid.uuid4()}"
    call_id_2 = f"call_{uuid.uuid4()}"

    # 1. User: "Check weather in London and Paris"
    # 2. Assistant: Calls get_weather for both
    # 3. Tool 1: Result London
    # 4. Tool 2: Result Paris

    messages = [
        OpenAIChatMessage(role="user", content="Check weather in London and Paris"),
        OpenAIChatMessage(
            role="assistant",
            tool_calls=[
                OpenAIToolCall(
                    id=call_id_1,
                    type="function",
                    function={"name": "get_weather", "arguments": '{"location": "London"}'},
                ),
                OpenAIToolCall(
                    id=call_id_2,
                    type="function",
                    function={"name": "get_weather", "arguments": '{"location": "Paris"}'},
                ),
            ],
        ),
        OpenAIChatMessage(
            role="tool",
            tool_call_id=call_id_1,
            content=json.dumps({"temperature": "15C", "location": "London"}),
        ),
        OpenAIChatMessage(
            role="tool",
            tool_call_id=call_id_2,
            content=json.dumps({"temperature": "20C", "location": "Paris"}),
        ),
    ]

    print("Input OpenAI Messages:")
    for m in messages:
        print(f"[{m.role}] ID: {getattr(m, 'tool_call_id', 'N/A')} Content: {m.content or 'Tool Calls'}")

    gemini_contents = _process_messages(messages)

    print("\nOutput Gemini Contents Count:", len(gemini_contents))

    # Debug output
    # print(json.dumps(gemini_contents, indent=2))

    # Expectation:
    # 0: User (text)
    # 1: Model (function calls)
    # 2: User (function responses - combined)
    assert len(gemini_contents) == 3, f"Expected 3 Gemini messages, got {len(gemini_contents)}"

    # Check the last message
    tool_response_msg = gemini_contents[2]
    assert tool_response_msg["role"] == "user" or tool_response_msg["role"] == "function"
    assert len(tool_response_msg["parts"]) == 2, f"Expected 2 parts in tool response, got {len(tool_response_msg['parts'])}"

    part1 = tool_response_msg["parts"][0]
    part2 = tool_response_msg["parts"][1]

    assert "functionResponse" in part1
    assert "functionResponse" in part2

    print("Test Passed: Multiple tool responses grouped into single message.")


if __name__ == "__main__":
    try:
        test_multi_tool_call_grouping()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
