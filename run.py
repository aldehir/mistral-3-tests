import json
import sys
from pathlib import Path
from openai import OpenAI
from typing import Any, List
from datetime import datetime, timedelta

from huggingface_hub import hf_hub_download
from openai.types.chat import ChatCompletionToolUnionParam

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 262144

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

#models = client.models.list()
model = "mistralai/Devstral-Small-2-24B-Instruct-2512"


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


SYSTEM_PROMPT = load_system_prompt(model, "CHAT_SYSTEM_PROMPT.txt")


def add_number(a: float | str, b: float | str) -> float:
    a, b = float(a), float(b)
    return a + b


def multiply_number(a: float | str, b: float | str) -> float:
    a, b = float(a), float(b)
    return a * b


def substract_number(a: float | str, b: float | str) -> float:
    a, b = float(a), float(b)
    return a - b


def write_a_story() -> str:
    return "A long time ago in a galaxy far far away..."


def terminal(command: str, args: dict[str, Any] | str) -> str:
    return "found nothing"


def python(code: str, result_variable: str) -> str:
    data = {}
    exec(code, data)
    return str(data[result_variable])


MAP_FN = {
    "add_number": add_number,
    "multiply_number": multiply_number,
    "substract_number": substract_number,
    "write_a_story": write_a_story,
    "terminal": terminal,
    "python": python,
}


# Load test script that defines messages and tools
if len(sys.argv) < 2:
    print("Usage: python test.py <path_to_test_script>")
    sys.exit(1)

test_script_path = sys.argv[1]
with open(test_script_path, "r") as f:
    test_script_code = f.read()

messages = []
tools: List[ChatCompletionToolUnionParam] = []

# Execute the test script in the current global context
# The script should define 'messages' and 'tools' variables
exec(test_script_code, globals())

# Add system prompt to messages if not already present
if not any(msg.get("role") == "system" for msg in messages):
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})


has_tool_calls = True
origin_messages_len = len(messages)
while has_tool_calls:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
        tools=tools if tools else [],
        tool_choice="auto" if tools else None,
    )
    tool_calls = response.choices[0].message.tool_calls
    content = response.choices[0].message.content
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [tc.to_dict() for tc in tool_calls]
            if tool_calls
            else tool_calls,
            "content": content,
        }
    )
    results = []
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            result = MAP_FN[function_name](**json.loads(function_args))
            results.append(result)
        for tool_call, result in zip(tool_calls, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": str(result),
                }
            )
    else:
        has_tool_calls = False

# Prepare output
output = json.dumps(messages[origin_messages_len:], indent=2)

# Print to stdout
print(output)

# Save to log file with same basename as test script but with .log suffix
log_file_path = Path(test_script_path).with_suffix('.log')
with open(log_file_path, 'w') as log_file:
    log_file.write(output)
