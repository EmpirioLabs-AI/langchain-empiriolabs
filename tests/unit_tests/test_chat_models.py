"""Unit tests for ChatEmpirioLabs.

These run without network access. Set a dummy key so the OpenAI client
constructs cleanly. For live behavior, the langchain standard test suite
(``langchain-tests``) provides ChatModelIntegrationTests; wire those up under
``tests/integration_tests/`` with a real ``EMPIRIOLABS_API_KEY`` when ready.
"""

import os

from langchain_empiriolabs import ChatEmpirioLabs
from langchain_empiriolabs.chat_models import DEFAULT_API_BASE, DEFAULT_MODEL

os.environ.setdefault("EMPIRIOLABS_API_KEY", "test-key")


def test_default_model_and_base_url() -> None:
    llm = ChatEmpirioLabs()
    assert llm.model_name == DEFAULT_MODEL
    assert llm.empiriolabs_api_base == DEFAULT_API_BASE
    assert llm._llm_type == "empiriolabs-chat"


def test_model_alias() -> None:
    llm = ChatEmpirioLabs(model="deepseek-v4-pro")
    assert llm.model_name == "deepseek-v4-pro"


def test_api_key_is_secret() -> None:
    llm = ChatEmpirioLabs(api_key="sk-secret", model="glm-5-1")
    # The key must not be printed in plain text.
    assert "sk-secret" not in repr(llm)
    assert llm.empiriolabs_api_key is not None
    assert llm.empiriolabs_api_key.get_secret_value() == "sk-secret"


def test_base_url_override() -> None:
    llm = ChatEmpirioLabs(base_url="https://api.empiriolabs.ai/v1/")
    assert llm.empiriolabs_api_base == "https://api.empiriolabs.ai/v1/"


def test_ls_params() -> None:
    llm = ChatEmpirioLabs(model="kimi-k2-7-code")
    params = llm._get_ls_params()
    assert params["ls_provider"] == "empiriolabs"
    assert params["ls_model_name"] == "kimi-k2-7-code"


def test_namespace_and_secrets() -> None:
    assert ChatEmpirioLabs.get_lc_namespace() == [
        "langchain_empiriolabs",
        "chat_models",
    ]
    llm = ChatEmpirioLabs(api_key="sk-secret")
    # The api key is registered as a serializable secret reference.
    assert llm.lc_secrets == {"empiriolabs_api_key": "EMPIRIOLABS_API_KEY"}
