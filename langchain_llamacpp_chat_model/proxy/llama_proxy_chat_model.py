from typing import Any, Dict
from llama_cpp.server.app import LlamaProxy

from langchain_llamacpp_chat_model.llama_chat_model import LlamaChatModel


class LlamaProxyChatModel(LlamaChatModel):
    def __init__(
        self,
        llama_proxy: LlamaProxy,
        **kwargs,
    ):
        model = kwargs.get("model_name", kwargs.get("model"))
        llama = llama_proxy(model)
        super().__init__(**kwargs, llama=llama)

    @property
    def _llm_type(self) -> str:
        return "llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
