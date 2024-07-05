# Langchain Chat Model + LLamaCPP

A working solution for Integrating LLamaCPP with langchain

## Usage

```python
import os
from llama_cpp.server.app import LlamaProxy
from llama_cpp.server.settings import ModelSettings

model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

settings = ModelSettings(
    model=model_path,
    model_alias="llama3",
    n_gpu_layers=-1, # Use GPU
    n_ctx=1024,
    n_batch=512,  # Should be between 1 and n_ctx, consider the amount of RAM
    offload_kqv=True,  # Equivalent of f16_kv=True
    chat_format="chatml-function-calling",
    verbose=False,
)

self.llama_proxy = LlamaProxy(models=[settings])

chat_model = LlamaCppChatModel(llama_proxy=self.llama_proxy, model_name=self.model_alias)

chat_model.invoke("Tell me a joke")
chat_model.stream("Tell me a joke")
```
