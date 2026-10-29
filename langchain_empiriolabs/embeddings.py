"""EmpirioLabs embeddings."""

from __future__ import annotations

from typing import Optional

from langchain_core.utils import from_env, secret_from_env
from langchain_openai import OpenAIEmbeddings
from pydantic import Field, SecretStr

DEFAULT_API_BASE = "https://api.empiriolabs.ai/v1"
DEFAULT_MODEL = "text-embedding-v4"


class EmpirioLabsEmbeddings(OpenAIEmbeddings):
    """EmpirioLabs embedding model integration.

    Setup:
        Install ``langchain-empiriolabs`` and set the environment variable
        ``EMPIRIOLABS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-empiriolabs
            export EMPIRIOLABS_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_empiriolabs import EmpirioLabsEmbeddings

            embeddings = EmpirioLabsEmbeddings(model="MODEL_ID")

    Embed single text:
        .. code-block:: python

            embeddings.embed_query("Hello, world!")

    Embed multiple texts:
        .. code-block:: python

            embeddings.embed_documents(["Hello, world!", "Goodbye, world!"])
    """

    model: str = Field(default=DEFAULT_MODEL)
    """The EmpirioLabs embedding model slug to call."""

    empiriolabs_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("EMPIRIOLABS_API_KEY", default=None),
    )
    """EmpirioLabs API key, read from ``EMPIRIOLABS_API_KEY`` if not provided."""

    empiriolabs_api_base: str = Field(
        alias="base_url",
        default_factory=from_env("EMPIRIOLABS_API_BASE", default=DEFAULT_API_BASE),
    )
    """Base URL for the EmpirioLabs API."""

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map of constructor arg name to secret id."""
        return {"empiriolabs_api_key": "EMPIRIOLABS_API_KEY"}

    def __init__(self, **kwargs: object) -> None:
        # OpenAIEmbeddings reads OPENAI_API_KEY / OPENAI_API_BASE under the hood,
        # so mirror the EmpirioLabs values onto the fields the parent expects.
        api_key = kwargs.get("api_key") or kwargs.get("empiriolabs_api_key")
        if api_key is not None:
            kwargs.setdefault("openai_api_key", api_key)
        base_url = (
            kwargs.get("base_url")
            or kwargs.get("empiriolabs_api_base")
            or from_env("EMPIRIOLABS_API_BASE", default=DEFAULT_API_BASE)()
        )
        kwargs.setdefault("openai_api_base", base_url)
        if kwargs.get("openai_api_key") is None:
            resolved = secret_from_env("EMPIRIOLABS_API_KEY", default=None)()
            if resolved is not None:
                kwargs["openai_api_key"] = resolved
        # text-embedding-v4 is not an OpenAI model, so tiktoken cannot map it.
        kwargs.setdefault("tiktoken_enabled", False)
        kwargs.setdefault("check_embedding_ctx_length", False)
        super().__init__(**kwargs)  # type: ignore[arg-type]
