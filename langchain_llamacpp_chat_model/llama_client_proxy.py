from llama_cpp import Llama


class LlamaCreateContextManager:

    def __init__(self, llama: Llama, **kwargs):
        self.llama = llama
        self.kwargs = kwargs
        self.response = None

    def __call__(self):
        self.kwargs.pop("n", None)
        self.kwargs.pop(
            "parallel_tool_calls", None
        )  # LLamaCPP does not support parallel tool calls for now

        self.response = self.llama.create_chat_completion(**self.kwargs)
        return self.response

    def __enter__(self):
        return self()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if hasattr(self.response, "close"):
            self.response.close()
        return False


class LLamaOpenAIClientProxy:
    def __init__(self, llama: Llama):
        self.llama = llama

    def create(self, **kwargs):
        proxy = LlamaCreateContextManager(llama=self.llama, **kwargs)
        if "stream" in kwargs and kwargs["stream"] is True:
            return proxy
        else:
            return proxy()
