# Langchain Chat Model + LLamaCPP

A well tested 🧪 working solution for Integrating LLamaCPP with langchain. Fully compatible with **ChatModel**, and **LangGraph integration**. Provide a direct interface to the LlamaCPP library, **without any additional wrapper layers**, to maintain full configurability and control over the LlamaCPP functionality.

If you find this project useful, please give it a star ⭐!

### Support:

- ✅ `invoke`
- ✅ `ainvoke`
- ✅ `stream`
- ✅ `astream`
- ✅ Structured output (JSON mode)
- ✅ Tool/Function calling
- ✅ LLamaProxy

## Quick Install

##### pip

```bash
pip install langchain-llamacpp-chat-model
```

```bash
# When using llama_proxy
pip install langchain-llamacpp-chat-model langchain-llamacpp-chat-model[llama_proxy]
```

##### poetry

```bash
poetry add langchain-llamacpp-chat-model
```

```bash
# When using llama_proxy
poetry add langchain-llamacpp-chat-model langchain-llamacpp-chat-model[llama_proxy]
```

## Usage

### Using Llama

Llama instance allow to create a chat model for a single llama model

```python
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from llama_cpp import Llama
from langchain_llamacpp_chat_model import LlamaChatModel
from langchain_core.tools import tool

model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

llama = Llama(
    model_path=model_path,
    _gpu_layers=-1,
    chat_format="chatml-function-calling", # https://llama-cpp-python.readthedocs.io/en/latest/#function-calling
)
chat_model = LlamaChatModel(llama=llama)
```

#### Invoke

```python
result = chat_model.invoke("Tell me a joke about cats")
print(
    result.content
)  # Why was the cat sitting on the computer? Because it wanted to keep an eye on the mouse!
```

#### Stream

```python
stream = chat_model.stream("Tell me a joke about cats")
final_content = ""
for token in stream:
    final_content += token.content

print(
    final_content
)  # Why was the cat sitting on the computer? Because it wanted to keep an eye on the mouse!

```

#### Strucuted Output

```python
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


structured_llm = chat_model.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a joke about cats")

assert isinstance(result, Joke)
print(result.setup)  # Why was the cat sitting on the computer?
print(result.punchline)  # Because it wanted to keep an eye on the mouse!
```
⚠️ Ensure temperature is set to 0 (or near 0). The open-source models (tested with Phi3 and LLama3) do not perform well when calling functions, as their behavior can be unpredictable.

#### Function calling

```python
@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


llm_with_tool = chat_model.bind_tools(
    [magic_number_tool], tool_choice="magic_number_tool"
)

result = llm_with_tool.invoke("What is the magic mumber of 2?")

assert result.tool_calls[0]["name"] == "magic_number_tool"
```
⚠️ Ensure temperature is set to 0 (or near 0). The open-source models (tested with Phi3 and LLama3) do not perform well when calling functions, as their behavior can be unpredictable.

### Using LlamaProxy

LLamaProxy allow to define multiple models and use one of them by specifying `model_name`. Very useful for a server environment.

```python
import os
from llama_cpp.server.app import LlamaProxy, ModelSettings
from langchain_llamacpp_chat_model import LlamaProxyChatModel

llama3_model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)
phi3_model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf",
)

llama_proxy = LlamaProxy(
    models=[
        ModelSettings(model=llama3_model_path, model_alias="llama3"),
        ModelSettings(model=phi3_model_path, model_alias="phi3"),
    ]
)
llama3_chat_model = LlamaProxyChatModel(llama_proxy=llama_proxy, model="llama3")
phi3_chat_model = LlamaProxyChatModel(llama_proxy=llama_proxy, model="phi3")


# Invoke
# --------------------------------------------------------
llama3_result = llama3_chat_model.invoke("Tell me a joke about cats")
print(llama3_result.content)

phi3_result = llama3_chat_model.invoke("Tell me a joke about cats")
print(phi3_result.content)

# Stream
# --------------------------------------------------------
stream = llama3_chat_model.stream("Tell me a joke about cats")
final_content = ""
for token in stream:
    final_content += token.content

print(final_content)
```
