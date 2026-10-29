# langchain-empiriolabs

This package contains the LangChain integration for [EmpirioLabs AI](https://empiriolabs.ai).

EmpirioLabs AI is specialized AI model hosting for open, proprietary, and custom stacks. Every model it serves is listed, with its ID and price, in the live catalog at [empiriolabs.ai/models](https://empiriolabs.ai/models).

## Installation

```bash
pip install -U langchain-empiriolabs
```

Set your API key as an environment variable:

```bash
export EMPIRIOLABS_API_KEY="your-api-key"
```

You can create a key from the [EmpirioLabs AI dashboard](https://platform.empiriolabs.ai).

## Chat models

`ChatEmpirioLabs` wraps the EmpirioLabs chat completions API.

```python
from langchain_empiriolabs import ChatEmpirioLabs

llm = ChatEmpirioLabs(model="MODEL_ID")  # any chat model ID from empiriolabs.ai/models

response = llm.invoke("Explain backpropagation in one paragraph.")
print(response.content)
```

Streaming:

```python
for chunk in llm.stream("Write a haiku about the sea."):
    print(chunk.content, end="", flush=True)
```

Tool calling:

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

llm_with_tools = llm.bind_tools([GetWeather])
print(llm_with_tools.invoke("What is the weather in SF?").tool_calls)
```

### Available models

Any chat model ID from the live catalog works: [empiriolabs.ai/models](https://empiriolabs.ai/models), or `GET https://api.empiriolabs.ai/v1/models` from code. The catalog changes as models are added and retired, so this page names none.

Reasoning models place their chain-of-thought in `response.additional_kwargs["reasoning_content"]`.


## Embeddings

`EmpirioLabsEmbeddings` wraps the EmpirioLabs embeddings API.

```python
from langchain_empiriolabs import EmpirioLabsEmbeddings

embeddings = EmpirioLabsEmbeddings(model="MODEL_ID")  # any embedding model ID from empiriolabs.ai/models

vector = embeddings.embed_query("Hello, world!")
print(len(vector))
```

A request takes up to 2,048 texts; `embed_documents` sends 1,000 per request by default (`chunk_size`).

## Configuration

| Argument | Environment variable | Default |
| --- | --- | --- |
| `api_key` | `EMPIRIOLABS_API_KEY` | (required) |
| `base_url` | `EMPIRIOLABS_API_BASE` | `https://api.empiriolabs.ai/v1` |

## License

MIT
