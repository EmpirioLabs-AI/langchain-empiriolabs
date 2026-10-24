"""LangChain integration for EmpirioLabs.

EmpirioLabs AI is specialized AI model hosting for open, proprietary, and
custom stacks: chat models such as Qwen3, DeepSeek V4, GLM-5.1, Kimi K2.7
Code and MiniMax M3, and embeddings.
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
