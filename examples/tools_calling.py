import os
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
    chat_format="chatml-function-calling",  # https://llama-cpp-python.readthedocs.io/en/latest/#function-calling
)
chat_model = LlamaChatModel(llama=llama)


# Funtion calling
# --------------------------------------------------------
@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


llm_with_tool = chat_model.bind_tools(
    [magic_number_tool], tool_choice="magic_number_tool"
)

result = llm_with_tool.invoke("What is the magic mumber of 2?")

assert result.tool_calls[0]["name"] == "magic_number_tool"
