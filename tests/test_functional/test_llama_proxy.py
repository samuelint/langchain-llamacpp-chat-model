import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_llamacpp_chat_model import LlamaProxyChatModel

from tests.test_functional.models_configuration import (
    create_llama_proxy,
    models_to_test,
)
from llama_cpp.server.app import LlamaProxy


@pytest.fixture
def llama_proxy() -> LlamaProxy:
    return create_llama_proxy()


class TestLlamaProxyChat:

    @pytest.fixture(
        params=models_to_test, ids=[config["alias"] for config in models_to_test]
    )
    def instance(self, llama_proxy: LlamaProxy, request):
        return LlamaProxyChatModel(
            llama_proxy=llama_proxy, model_name=request.param["alias"]
        )

    def test_conversation_memory(self, instance: LlamaProxyChatModel):
        result = instance.invoke(
            input=[
                HumanMessage(content="Remember that I like bananas"),
                AIMessage(content="Okay"),
                HumanMessage(content="What do I like?"),
            ]
        )

        assert "banana" in result.content
