from pydantic import Field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.pydantic_v1 import BaseModel

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessageChunk, BaseMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from llama_cpp import (
    CreateCompletionResponse,
    CreateCompletionStreamResponse,
    Literal,
    LlamaGrammar,
    LogitsProcessorList,
    StoppingCriteriaList,
    Type,
    Union,
)
from llama_cpp.server.app import LlamaProxy

from langchain_llamacpp_chat_model.llama_chat_model import LlamaChatModel

# Use this class until it's implemented in LangChain Community


class LlamaProxyChatModel(LlamaChatModel):
    model_name: str = Field(default="", alias="model")

    suffix: Optional[str] = None
    max_tokens: Optional[int] = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.0
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = []
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.1
    top_k: int = 40
    seed: Optional[int] = None
    tfs_z: float = 1.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    stopping_criteria: Optional[StoppingCriteriaList] = None
    logits_processor: Optional[LogitsProcessorList] = None
    grammar: Optional[LlamaGrammar] = None
    logit_bias: Optional[Dict[str, float]] = None

    def __init__(
        self,
        llama_proxy: LlamaProxy,
        **kwargs,
    ):
        llama = llama_proxy(self.model_name)
        super().__init__(**kwargs, llama=llama)

    @property
    def _llm_type(self) -> str:
        return "llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
