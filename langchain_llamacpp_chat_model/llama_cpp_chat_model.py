from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from llama_cpp import (
    CreateCompletionResponse,
    CreateCompletionStreamResponse,
    LlamaGrammar,
    LogitsProcessorList,
    StoppingCriteriaList,
    Union,
)
from llama_cpp.server.app import LlamaProxy


class LlamaCppChatModel(BaseChatModel):
    llama_proxy: LlamaProxy
    model_name: str

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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message: CreateCompletionResponse = self._create_completion(
            messages=messages,
            stop=stop,
        )

        content = message["choices"][0]["text"]
        usage_metadata = {
            "input_tokens": message["usage"]["prompt_tokens"],
            "output_tokens": message["usage"]["completion_tokens"],
            "total_tokens": message["usage"]["total_tokens"],
        }
        generation = ChatGeneration(
            message=AIMessage(content=content, usage_metadata=usage_metadata),
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "id": message["id"],
                "model": message["model"],
                "created": message["created"],
            },
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        stream = self._create_completion(messages=messages, stop=stop, stream=True)

        for token in stream:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=token["choices"][0]["text"],
                    response_metadata={
                        "id": token["id"],
                        "created": token["created"],
                    },
                ),
            )
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

    def _to_string_prompt(self, messages: List[BaseMessage]) -> str:
        return "\n".join([message.content for message in messages])

    def _create_completion(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]:
        llama = self.llama_proxy(self.model_name)
        return llama.create_completion(
            prompt=self._to_string_prompt(messages),
            stop=stop or self.stop,
            stream=stream,
            suffix=self.suffix,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            typical_p=self.typical_p,
            logprobs=self.logprobs,
            echo=self.echo,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repeat_penalty=self.repeat_penalty,
            top_k=self.top_k,
            seed=self.seed,
            tfs_z=self.tfs_z,
            mirostat_mode=self.mirostat_mode,
            mirostat_tau=self.mirostat_tau,
            mirostat_eta=self.mirostat_eta,
            stopping_criteria=self.stopping_criteria,
            logits_processor=self.logits_processor,
            grammar=self.grammar,
            logit_bias=self.logit_bias,
        )

    @property
    def _llm_type(self) -> str:
        return "llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
