from llama_cpp import Llama

from langchain_openai.chat_models.base import BaseChatOpenAI

from .llama_client_proxy import LLamaOpenAIClientProxy
from .llama_client_async_proxy import LLamaOpenAIClientAsyncProxy


class LlamaChatModel(BaseChatOpenAI):
    model_name: str = "unknown"
    llama: Llama = None

    def __init__(
        self,
        llama: Llama,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            client=LLamaOpenAIClientProxy(llama=llama),
            async_client=LLamaOpenAIClientAsyncProxy(llama=llama),
        )
        self.llama = llama

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return self.llama.model_path
