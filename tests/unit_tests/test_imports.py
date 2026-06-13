"""Verify the public API surface of langchain_empiriolabs."""

import langchain_empiriolabs

EXPECTED_ALL = [
    "ChatEmpirioLabs",
    "EmpirioLabsEmbeddings",
    "__version__",
]


def test_all_imports() -> None:
    assert sorted(langchain_empiriolabs.__all__) == sorted(EXPECTED_ALL)
