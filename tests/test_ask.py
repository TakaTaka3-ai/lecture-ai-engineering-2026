from types import SimpleNamespace

import pytest

from app import ask as ask_module


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_token_count": 12, "candidates_token_count": 5},
        SimpleNamespace(prompt_token_count=12, candidates_token_count=5),
    ],
)
def test_ask_prints_usage_metadata(monkeypatch, capsys, usage):
    response = SimpleNamespace(text="answer", usage_metadata=usage)

    class FakeModels:
        def generate_content(self, **kwargs):
            return response

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("question") == "answer"
    assert "[usage] input_tokens=12 output_tokens=5" in capsys.readouterr().out


def test_ask_prints_zero_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata={"prompt_token_count": 0, "candidates_token_count": 0},
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return response

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    ask_module.ask("question")
    assert "[usage] input_tokens=0 output_tokens=0" in capsys.readouterr().out