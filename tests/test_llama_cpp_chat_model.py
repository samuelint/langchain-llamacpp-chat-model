import pprint
from unittest.mock import MagicMock

from llama_cpp import Llama
import pytest
from langchain_llamacpp_chat_model.llama_cpp_chat_model import LlamaCppChatModel
from llama_cpp.server.app import LlamaProxy
from langchain_core.messages import HumanMessage


@pytest.fixture
def model_name():
    return "test-model"


@pytest.fixture
def llama_mock():
    return MagicMock(spec=Llama)


@pytest.fixture
def llama_proxy_mock(llama_mock):
    mock = MagicMock(spec=LlamaProxy)
    mock.side_effect = lambda *args, **kwargs: llama_mock

    return mock


@pytest.fixture
def instance(llama_proxy_mock, model_name):
    return LlamaCppChatModel(llama_proxy=llama_proxy_mock, model_name=model_name)


class TestLlamaCppChatModelGenerate:

    def test_generate_content(self, instance: LlamaCppChatModel, llama_mock):
        messages = [HumanMessage(content="Hello, how are you?")]
        expected_response = {
            "id": "test-id",
            "model": "test-model",
            "created": 1234567890,
            "choices": [{"text": "I'm doing well, thank you!"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        llama_mock.create_completion.return_value = expected_response
        result = instance._generate(messages=messages)

        assert result.generations[0].message.content == "I'm doing well, thank you!"

    def test_generate_usage(self, instance: LlamaCppChatModel, llama_mock):
        messages = [HumanMessage(content="Hello, how are you?")]
        expected_response = {
            "id": "test-id",
            "model": "test-model",
            "created": 1234567890,
            "choices": [{"text": "I'm doing well, thank you!"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        llama_mock.create_completion.return_value = expected_response
        result = instance._generate(messages=messages)

        pprint.pprint(result)
        assert result.generations[0].message.usage_metadata["input_tokens"] == 10
        assert result.generations[0].message.usage_metadata["output_tokens"] == 5
        assert result.generations[0].message.usage_metadata["total_tokens"] == 15

    def test_generate_metadata(self, instance: LlamaCppChatModel, llama_mock):
        messages = [HumanMessage(content="Hello, how are you?")]
        expected_response = {
            "id": "test-id",
            "model": "test-model",
            "created": 1234567890,
            "choices": [{"text": "I'm doing well, thank you!"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        llama_mock.create_completion.return_value = expected_response
        result = instance._generate(messages=messages)

        assert result.llm_output["id"] == "test-id"
        assert result.llm_output["model"] == "test-model"
        assert result.llm_output["created"] == 1234567890


class TestLlamaCppChatModelStream:

    def test_stream(self, instance: LlamaCppChatModel, llama_mock):
        messages = [HumanMessage(content="Hello, how are you?")]
        expected_response = [
            {
                "id": "test-id-1",
                "created": 1234567890,
                "choices": [{"text": "I'm "}],
            },
            {
                "id": "test-id-2",
                "created": 1234567891,
                "choices": [{"text": "doing "}],
            },
            {
                "id": "test-id-3",
                "created": 1234567892,
                "choices": [{"text": "well, "}],
            },
            {
                "id": "test-id-4",
                "created": 1234567893,
                "choices": [{"text": "thank you!"}],
            },
        ]

        llama_mock.create_completion.return_value = iter(expected_response)
        chunks = list(instance._stream(messages))

        assert len(chunks) == 4
        assert chunks[0].message.content == "I'm "
        assert chunks[0].message.response_metadata["id"] == "test-id-1"
        assert chunks[0].message.response_metadata["created"] == 1234567890
        assert chunks[1].message.content == "doing "
        assert chunks[1].message.response_metadata["id"] == "test-id-2"
        assert chunks[1].message.response_metadata["created"] == 1234567891
        assert chunks[2].message.content == "well, "
        assert chunks[2].message.response_metadata["id"] == "test-id-3"
        assert chunks[2].message.response_metadata["created"] == 1234567892
        assert chunks[3].message.content == "thank you!"
        assert chunks[3].message.response_metadata["id"] == "test-id-4"
        assert chunks[3].message.response_metadata["created"] == 1234567893
