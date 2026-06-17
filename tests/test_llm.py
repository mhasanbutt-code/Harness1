from harness.config import HarnessConfig
from harness.llm.data import Example, read_jsonl
from harness.llm.eval import evaluate, token_f1
from harness.llm.model import LocalLLM


def test_read_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"instruction": "hi", "response": "hello"}\n')
    examples = read_jsonl(p)
    assert len(examples) == 1
    assert examples[0].instruction == "hi"
    assert "### Instruction:" in examples[0].prompt()


def test_read_jsonl_validates(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"instruction": "no response"}\n')
    try:
        read_jsonl(p)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_echo_llm_generates():
    llm = LocalLLM(HarnessConfig()).load()
    assert llm.kind in {"echo", "transformers"}
    out = llm.generate("### Instruction:\nGOAL: say hi\n### Response:\n")
    assert isinstance(out, str) and out


def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b c", "x y z") == 0.0
    assert 0 < token_f1("a b c", "a b x") < 1


def test_evaluate_runs():
    examples = [Example("q", "a"), Example("q2", "a2")]
    result = evaluate(examples, config=HarnessConfig())
    assert result.n == 2
    assert 0.0 <= result.token_f1 <= 1.0
