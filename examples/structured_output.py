import os
from langchain_core.pydantic_v1 import BaseModel, Field
from llama_cpp import Llama
from langchain_llamacpp_chat_model import LlamaChatModel

model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

llama = Llama(
    model_path=model_path, _gpu_layers=-1, chat_format="chatml-function-calling"
)
chat_model = LlamaChatModel(llama=llama)


# Strucuted Output
# --------------------------------------------------------
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


structured_llm = chat_model.with_structured_output(Joke)

result = structured_llm.invoke("Tell me a joke about cats")

assert isinstance(result, Joke)
print(result.setup)  # Why was the cat sitting on the computer?
print(result.punchline)  # Because it wanted to keep an eye on the mouse!
