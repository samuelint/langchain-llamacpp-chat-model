import os
from llama_cpp import Llama

models_to_test = [
    {
        "repo_id": "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    },
    {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
    },
]


def create_llama(request) -> Llama:
    local_path = os.path.join(
        os.path.expanduser("~/.cache/lm-studio/models"),
        request.param["repo_id"],
        request.param["filename"],
    )

    return Llama(
        model_path=local_path,
        n_gpu_layers=-1,
    )
