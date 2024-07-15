from llama_cpp import Llama
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_llamacpp_chat_model import LlamaChatModel
from tests.test_functional.models_configuration import create_llama, models_to_test


class TestStream:

    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request.param)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama)

    def test_stream(self, instance: LlamaChatModel):
        stream = instance.stream("Say Hi!")

        final_content = ""
        for token in stream:
            final_content += token.content

        assert len(final_content) > 0

    def test_response_do_not_start_with_2_endl(self, instance: LlamaChatModel):
        stream = instance.stream("Say Hi!")

        final_content = ""
        for token in stream:
            final_content += token.content

        assert final_content.startswith("\n\n") is False

    def test_conversation_memory(self, instance: LlamaChatModel):
        stream = instance.stream(
            input=[
                HumanMessage(content="Remember that I like bananas"),
                AIMessage(content="Okay"),
                HumanMessage(content="What do I like?"),
            ]
        )

        final_content = ""
        for token in stream:
            final_content += token.content

        assert len(final_content) > 0
        assert "banana" in final_content
