from llama_cpp import Llama
import pytest
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_llamacpp_chat_model import LlamaChatModel
from langchain_core.messages import BaseMessage
from tests.test_functional.models_configuration import create_llama, models_to_test


class TestSystemMessage:

    @pytest.fixture(
        params=models_to_test,
        ids=[config["repo_id"] for config in models_to_test],
        scope="session",
    )
    def llama(self, request) -> Llama:
        return create_llama(request.param)

    @pytest.fixture(scope="session")
    def instance(self, llama):
        return LlamaChatModel(llama=llama, temperature=0)

    @pytest.fixture(scope="session")
    def result(self, instance: LlamaChatModel):
        return instance.invoke(
            input=[
                SystemMessage(
                    content="Answer like a pirate. Finish every sentence with a arrr"
                ),
                HumanMessage(content="Remember that I like bananas"),
            ]
        )

    def test_system_message_is_taken_in_account(self, result: BaseMessage):
        assert "arrr" in result.content.lower()

    def test_response_does_not_contains_im_start(self, result: BaseMessage):
        assert "<|im_start|>" not in result.content.lower()
