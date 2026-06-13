"""EmpirioLabs chat models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_API_BASE = "https://api.empiriolabs.ai/v1"
DEFAULT_MODEL = "qwen3-7-plus"


class ChatEmpirioLabs(BaseChatOpenAI):
    """EmpirioLabs chat model integration.

    EmpirioLabs serves frontier open models through one OpenAI-compatible API.
    See the model catalog at https://docs.empiriolabs.ai for available slugs.

    Setup:
        Install ``langchain-empiriolabs`` and set the environment variable
        ``EMPIRIOLABS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-empiriolabs
            export EMPIRIOLABS_API_KEY="your-api-key"

    Key init args, completion params:
        model: str
            Name of the EmpirioLabs model to use, e.g. ``qwen3-7-plus``,
            ``deepseek-v4-pro``, ``glm-5-1``, or ``kimi-k2-7-code``.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args, client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            EmpirioLabs API key. If not passed in, it is read from the env var
            ``EMPIRIOLABS_API_KEY``.
        base_url: Optional[str]
            Endpoint to call. Defaults to ``https://api.empiriolabs.ai/v1``.

    Instantiate:
        .. code-block:: python

            from langchain_empiriolabs import ChatEmpirioLabs

            llm = ChatEmpirioLabs(
                model="qwen3-7-plus",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "Write a haiku about the sea."),
            ]
            llm.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="")

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location.'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )

            llm_with_tools = llm.bind_tools([GetWeather])
            llm_with_tools.invoke("What is the weather in SF?")

    Reasoning models (e.g. ``deepseek-v4-pro``, ``kimi-k2-7-code``) place their
    chain-of-thought in ``additional_kwargs["reasoning_content"]`` on the
    returned message.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """The EmpirioLabs model slug to call."""

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
    def lc_secrets(self) -> Dict[str, str]:
        """Map of constructor arg name to secret id."""
        return {"empiriolabs_api_key": "EMPIRIOLABS_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_empiriolabs", "chat_models"]

    # Note: we intentionally do NOT override ``is_lc_serializable`` to return
    # True. langchain-core's ``load()`` only deserializes classes in its own
    # built-in allowlist (first-party packages), so an out-of-tree partner
    # class cannot round-trip through ``load()`` on current langchain-core.
    # Leaving the default (False) is the standard partner-package behavior and
    # makes the standard ``test_serdes`` suite skip cleanly. ``dumpd`` still
    # works for inspection via ``lc_secrets`` / ``lc_attributes`` below.

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """Attribute names that should be included in the serialized kwargs."""
        attributes: Dict[str, Any] = {}
        if self.empiriolabs_api_base:
            attributes["empiriolabs_api_base"] = self.empiriolabs_api_base
        return attributes

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "empiriolabs"
        return params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate the config and build the OpenAI-compatible clients."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params: dict = {
            "api_key": (
                self.empiriolabs_api_key.get_secret_value()
                if self.empiriolabs_api_key
                else None
            ),
            "base_url": self.empiriolabs_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return self

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        if hasattr(response.choices[0].message, "reasoning_content"):  # type: ignore[union-attr]
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                response.choices[0].message.reasoning_content  # type: ignore[union-attr]
            )

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: Type,
        base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if reasoning_content := top.get("delta", {}).get("reasoning_content"):
                if isinstance(generation_chunk.message, AIMessageChunk):
                    generation_chunk.message.additional_kwargs[
                        "reasoning_content"
                    ] = reasoning_content
        return generation_chunk

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "empiriolabs-chat"
