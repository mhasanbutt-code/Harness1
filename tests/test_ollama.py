"""Ollama backend tests that don't require a running daemon."""

from harness.config import HarnessConfig
from harness.llm.model import LocalLLM
from harness.llm.ollama import OllamaLLM, ping, select_model


def test_select_model_exact_match():
    assert select_model("qwen3:32b", ["qwen3:32b", "llama3"]) == "qwen3:32b"


def test_select_model_same_family():
    # Requested tag absent, but a qwen3:* model is installed -> use it.
    assert select_model("qwen3:32b", ["qwen3:8b", "nomic-embed-text"]) == "qwen3:8b"


def test_select_model_skips_embedding_models():
    assert select_model("qwen3:32b", ["nomic-embed-text", "llama3:8b"]) == "llama3:8b"


def test_select_model_empty_keeps_requested():
    assert select_model("qwen3:32b", []) == "qwen3:32b"


def test_ping_unreachable_host_is_false():
    # Nothing is serving on this port in test environments.
    assert ping("http://127.0.0.1:1", timeout=1.0) is False


def test_ollama_available_false_when_down():
    assert OllamaLLM("http://127.0.0.1:1", "qwen3").available() is False


def test_ollama_option_mapping():
    ol = OllamaLLM()
    opts = ol._options({"temperature": 0.3, "max_new_tokens": 64})
    assert opts == {"temperature": 0.3, "num_predict": 64}


def test_localllm_auto_falls_back_to_echo_without_backends():
    # No Ollama, no transformers in the test env -> deterministic echo.
    llm = LocalLLM(HarnessConfig(llm_backend="auto", ollama_host="http://127.0.0.1:1")).load()
    assert llm.kind == "echo"
    assert "echo-model" in llm.generate("Question: hello\nAnswer:")


def test_localllm_explicit_echo():
    llm = LocalLLM(HarnessConfig(llm_backend="echo")).load()
    assert llm.kind == "echo"
