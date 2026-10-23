# langchain-empiriolabs

This package contains the LangChain integration for [EmpirioLabs AI](https://empiriolabs.ai).

EmpirioLabs AI is specialized AI model hosting for open, proprietary, and custom stacks. Its chat models include Qwen3, DeepSeek V4, GLM-5.1, Kimi K2.7 Code and MiniMax M3; browse the full model catalog at [docs.empiriolabs.ai](https://docs.empiriolabs.ai).

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

llm = ChatEmpirioLabs(model="qwen3-7-plus")

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

Any model slug from the EmpirioLabs AI catalog works. Some popular chat slugs:

| Slug | Notes |
| --- | --- |
| `qwen3-7-plus` | General-purpose chat (default) |
| `qwen3-7-max` | Highest-capability Qwen3 |
| `deepseek-v4-pro` | Reasoning model |
| `deepseek-v4-1-flash` | Fast |
| `glm-5-1` | GLM-5.1 |
| `kimi-k2-7-code` | Coding-focused, reasoning |
| `minimax-m3` | MiniMax M3 |

Reasoning models place their chain-of-thought in `response.additional_kwargs["reasoning_content"]`.

The live list is always available from `GET https://api.empiriolabs.ai/v1/models`.

## Embeddings

`EmpirioLabsEmbeddings` wraps the EmpirioLabs embeddings API.

```python
from langchain_empiriolabs import EmpirioLabsEmbeddings

embeddings = EmpirioLabsEmbeddings(model="text-embedding-v4")

vector = embeddings.embed_query("Hello, world!")
print(len(vector))
```

## Configuration

| Argument | Environment variable | Default |
| --- | --- | --- |
| `api_key` | `EMPIRIOLABS_API_KEY` | (required) |
| `base_url` | `EMPIRIOLABS_API_BASE` | `https://api.empiriolabs.ai/v1` |

## License

MIT
