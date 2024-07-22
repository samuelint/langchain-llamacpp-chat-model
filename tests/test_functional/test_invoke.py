from llama_cpp import Llama
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_llamacpp_chat_model import LlamaChatModel

from tests.test_functional.models_configuration import create_llama, models_to_test


class TestInvoke:

    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request.param)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama, temperature=0)

    def test_invoke(self, instance: LlamaChatModel):
        result = instance.invoke("Say Hi!")

        assert len(result.content) > 0

    def test_response_do_not_start_with_2_endl(self, instance: LlamaChatModel):
        result = instance.invoke("Say Hi!")

        assert result.content.startswith("\n\n") is False

    def test_conversation_memory(self, instance: LlamaChatModel):
        result = instance.invoke(
            input=[
                HumanMessage(content="Remember that I like bananas"),
                AIMessage(content="Okay"),
                HumanMessage(content="What do I like?"),
            ]
        )

        assert "banana" in result.content.lower()
