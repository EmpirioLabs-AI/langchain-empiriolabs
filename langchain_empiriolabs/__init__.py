"""LangChain integration for EmpirioLabs.

EmpirioLabs serves frontier open models (Qwen3, DeepSeek V4, GLM-5.1,
Kimi K2.7 Code, MiniMax M3, and more) through one OpenAI-compatible API.
"""

from importlib import metadata

from langchain_empiriolabs.chat_models import ChatEmpirioLabs
from langchain_empiriolabs.embeddings import EmpirioLabsEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Package is not installed (e.g. running from a source checkout).
    __version__ = ""
del metadata  # avoid polluting the namespace

__all__ = [
    "ChatEmpirioLabs",
    "EmpirioLabsEmbeddings",
    "__version__",
]
