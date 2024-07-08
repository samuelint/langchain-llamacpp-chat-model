import os
from llama_cpp import Llama
from langchain_llamacpp_chat_model import LlamaChatModel

model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

llama = Llama(
    model_path=model_path, _gpu_layers=-1, chat_format="chatml-function-calling"
)
chat_model = LlamaChatModel(llama=llama)

# Stream
# --------------------------------------------------------
stream = chat_model.stream("Tell me a joke about cats")
final_content = ""
for token in stream:
    final_content += token.content

print(
    final_content
)  # Why was the cat sitting on the computer? Because it wanted to keep an eye on the mouse!
