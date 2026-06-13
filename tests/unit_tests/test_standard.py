"""LangChain standard unit tests for ChatEmpirioLabs.

``langchain-tests`` ships ChatModelUnitTests (no network) and
ChatModelIntegrationTests (live calls). This wires up the unit suite. To add
the integration suite, create ``tests/integration_tests/test_chat_models.py``
with a ChatModelIntegrationTests subclass and run it with a real
``EMPIRIOLABS_API_KEY``.

See https://python.langchain.com/docs/contributing/how_to/integrations/standard_tests/
"""

from typing import Tuple, Type

import pytest

from langchain_empiriolabs import ChatEmpirioLabs

try:
    from langchain_tests.unit_tests import ChatModelUnitTests

    HAS_STANDARD_TESTS = True
except ImportError:  # pragma: no cover
    HAS_STANDARD_TESTS = False


@pytest.mark.skipif(
    not HAS_STANDARD_TESTS, reason="langchain-tests is not installed"
)
class TestChatEmpirioLabsUnit(ChatModelUnitTests):  # type: ignore[misc]
    @property
    def chat_model_class(self) -> Type[ChatEmpirioLabs]:
        return ChatEmpirioLabs

    @property
    def chat_model_params(self) -> dict:
        return {"model": "qwen3-7-plus"}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        # (env vars to set, init kwargs, expected attributes). Setting the key
        # via env lets the serdes round-trip revive the SecretStr from env.
        return (
            {
                "EMPIRIOLABS_API_KEY": "api_key",
                "EMPIRIOLABS_API_BASE": "https://api.empiriolabs.ai/v1",
            },
            {},
            {
                "empiriolabs_api_key": "api_key",
                "empiriolabs_api_base": "https://api.empiriolabs.ai/v1",
            },
        )
