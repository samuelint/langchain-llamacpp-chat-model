import multiprocessing
import os
from llama_cpp import Llama
from llama_cpp.server.app import LlamaProxy
from llama_cpp.server.settings import ModelSettings

models_to_test = [
    {
        "repo_id": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "alias": "llama3",
    },
    {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "alias": "phi3",
    },
]

n_gpu_layers = (
    -1  # -1 to offload on GPU. Until GPU is supported on Github, must be run on CPU
)


def _model_local_path(model) -> str:
    return os.path.join(
        os.path.expanduser("~/.cache/lm-studio/models"),
        model["repo_id"],
        model["filename"],
    )


def _create_models_settings():
    models: list[ModelSettings] = []
    for model in models_to_test:
        local_path = _model_local_path(model)
        models.append(
            ModelSettings(
                model=local_path,
                model_alias=model["alias"],
                n_gpu_layers=n_gpu_layers,
            )
        )

    return models


def create_llama(params) -> Llama:
    local_path = _model_local_path(params)

    return Llama(
        model_path=local_path,
        n_gpu_layers=n_gpu_layers,
        offload_kqv=True,  # Equivalent of f16_kv=True
        n_threads=multiprocessing.cpu_count() - 1,
        chat_format="chatml-function-calling",
    )


def create_llama_proxy() -> LlamaProxy:
    models = _create_models_settings()
    return LlamaProxy(models=models)
