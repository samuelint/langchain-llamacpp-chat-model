from llama_cpp import Llama
import pytest

from langchain_llamacpp_chat_model import LlamaChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from tests.test_functional.models_configuration import create_llama, models_to_test


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


class TestFunctionCalling:

    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request.param)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama, temperature=0)

    def test_json_mode(self, instance: LlamaChatModel):
        structured_llm = instance.with_structured_output(Joke)

        result = structured_llm.invoke("Tell me a joke about cats")

        assert isinstance(result, Joke)
        assert len(result.setup) > 0
        assert len(result.punchline) > 0

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

    def test_auto_function_calling(self, instance: LlamaChatModel):
        @tool
        def magic_number_tool(input: int) -> int:
            """Applies a magic function to an input."""
            return input + 2

        llm_with_tool = instance.bind_tools([magic_number_tool], tool_choice="auto")

        result = llm_with_tool.invoke(
            [
                SystemMessage(
                    content="The assistant calls functions with appropriate input when necessary."
                ),
                HumanMessage(content="What is the magic mumber of 2?"),
            ]
        )

        assert result.tool_calls[0]["name"] == "magic_number_tool"


class TestFunctionCallingWithStream:
    @pytest.fixture(
        params=models_to_test, ids=[config["repo_id"] for config in models_to_test]
    )
    def llama(self, request) -> Llama:
        return create_llama(request.param)

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama, temperature=0, streaming=True)

    def test_force_function_calling(self, instance: LlamaChatModel):
        @tool
        def magic_number_tool(input: int) -> int:
            """Applies a magic function to an input."""
            return input + 2

        llm_with_tool = instance.bind_tools(
            [magic_number_tool], tool_choice="magic_number_tool"
        )

        stream = llm_with_tool.stream(
            [
                HumanMessage(content="What is the magic mumber of 2?"),
            ]
        )

        tool_call_chunks = []
        for chunk in stream:
            if len(chunk.tool_call_chunks) > 0:
                tool_call_chunks.extend(chunk.tool_call_chunks)

        assert len(tool_call_chunks) > 0
        assert tool_call_chunks[0]["name"] == "magic_number_tool"

    @pytest.mark.skip(
        reason="""\
        Stream + auto tool choice not supported yet. \
        https://github.com/abetlen/llama-cpp-python/discussions/1615\
        """
    )
    def test_auto_function_calling(self, instance: LlamaChatModel):
        @tool
        def magic_number_tool(input: int) -> int:
            """Applies a magic function to an input."""
            return input + 2

        llm_with_tool = instance.bind_tools([magic_number_tool], tool_choice="auto")

        stream = llm_with_tool.stream(
            [
                HumanMessage(content="What is the magic mumber of 2?"),
            ]
        )

        tool_call_chunks = []
        for chunk in stream:
            if len(chunk.tool_call_chunks) > 0:
                tool_call_chunks.extend(chunk.tool_call_chunks)

        assert len(tool_call_chunks) > 0
        assert tool_call_chunks[0]["name"] == "magic_number_tool"
