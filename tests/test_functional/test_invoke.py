from llama_cpp import Llama
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_llamacpp_chat_model import LlamaChatModel

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from tests.test_functional.models_configuration import create_llama, models_to_test


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


class TestInvoke:

    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama)

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

        assert "banana" in result.content

    def test_json_mode(self, instance: LlamaChatModel):
        structured_llm = instance.with_structured_output(Joke)

        result = structured_llm.invoke("Tell me a joke about cats")

        assert isinstance(result, Joke)

    def test_force_function_calling(self, instance: LlamaChatModel):
        @tool
        def magic_number_tool(input: int) -> int:
            """Applies a magic function to an input."""
            return input + 2

        llm_with_tool = instance.bind_tools(
            [magic_number_tool], tool_choice="magic_number_tool"
        )

        result = llm_with_tool.invoke("What is the magic mumber of 2?")

        assert result.tool_calls[0]["name"] == "magic_number_tool"


class TestAInvoke:

    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama)

    @pytest.mark.asyncio
    async def test_ainvoke(self, instance: LlamaChatModel):
        result = await instance.ainvoke("Say Hi!")

        assert len(result.content) > 0
