from llama_cpp import Llama

from langchain_openai.chat_models.base import BaseChatOpenAI

from .llama_client_proxy import LLamaOpenAIClientProxy
from .llama_client_async_proxy import LLamaOpenAIClientAsyncProxy


class LlamaChatModel(BaseChatOpenAI):
    model_name: str = "unknown"

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
