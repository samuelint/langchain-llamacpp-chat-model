from llama_cpp import Llama
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_llamacpp_chat_model import LlamaChatModel
from tests.test_functional.models_configuration import create_llama, models_to_test


class TestAStream:

    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request.param)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama, temperature=0)

    @pytest.mark.asyncio
    async def test_astream(self, instance: LlamaChatModel):

        chunks = []
        async for chunk in instance.astream("Say Hi!"):
            chunks.append(chunk)

        final_content = "".join(chunk.content for chunk in chunks)

        assert len(final_content) > 0

    @pytest.mark.asyncio
    async def test_conversation_memory(self, instance: LlamaChatModel):
        stream = instance.astream(
            input=[
                HumanMessage(content="Remember that I like bananas"),
                AIMessage(content="Okay"),
                HumanMessage(content="What do I like?"),
            ]
        )

        final_content = ""
        async for token in stream:
            final_content += token.content

        assert len(final_content) > 0
        assert "banana" in final_content.lower()
