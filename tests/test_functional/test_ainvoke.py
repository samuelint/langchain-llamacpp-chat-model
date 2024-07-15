from llama_cpp import Llama
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_llamacpp_chat_model import LlamaChatModel

from langchain_core.pydantic_v1 import BaseModel, Field
from tests.test_functional.models_configuration import create_llama, models_to_test


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


class TestAInvoke:

    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request.param)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama)

    @pytest.mark.asyncio
    async def test_ainvoke(self, instance: LlamaChatModel):
        result = await instance.ainvoke("Say Hi!")

        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_conversation_memory(self, instance: LlamaChatModel):
        result = await instance.ainvoke(
            input=[
                HumanMessage(content="Remember that I like bananas"),
                AIMessage(content="Okay"),
                HumanMessage(content="What do I like?"),
            ]
        )

        assert "banana" in result.content
