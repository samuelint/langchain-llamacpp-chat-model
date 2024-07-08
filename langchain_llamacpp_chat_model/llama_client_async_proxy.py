from llama_cpp import Llama


async def to_async_iterator(iterator):
    for item in iterator:
        yield item


class LlamaCreateAsyncContextManager:

    def __init__(self, llama: Llama, **kwargs):
        self.llama = llama
        self.kwargs = kwargs
        self.response = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.response)
        except Exception:
            raise StopAsyncIteration()

    def __call__(self):
        self.kwargs.pop("n", None)
        self.kwargs.pop(
            "parallel_tool_calls", None
        )  # LLamaCPP does not support parallel tool calls for now

        self.response = self.llama.create_chat_completion(**self.kwargs)
        return self.response

    async def __aenter__(self):
        return self()

    async def __aexit__(self, exception_type, exception_value, exception_traceback):
        if hasattr(self.response, "close"):
            self.response.close()
        return False


class LLamaOpenAIClientAsyncProxy:
    def __init__(self, llama: Llama):
        self.llama = llama

    async def create(self, **kwargs):
        proxy = LlamaCreateAsyncContextManager(llama=self.llama, **kwargs)
        if "stream" in kwargs and kwargs["stream"] is True:
            return proxy
        else:
            return proxy()
