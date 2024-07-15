from llama_cpp import Llama
import pytest
from langchain_llamacpp_chat_model import LlamaChatModel
from tests.test_functional.models_configuration import create_llama, models_to_test


class TestInvoke:

    @pytest.fixture()
    def llama(self) -> Llama:

        return create_llama(models_to_test[0])

    @pytest.fixture
    def instance(self, llama):
        return LlamaChatModel(llama=llama)

    def test_llm_type(self, instance: LlamaChatModel):
        result = instance._llm_type
        assert models_to_test[0]["repo_id"] in result
