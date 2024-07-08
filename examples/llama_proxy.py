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
