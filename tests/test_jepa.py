import numpy as np

from harness.config import HarnessConfig
from harness.jepa.encoder import JEPAEncoder
from harness.jepa.interface import JEPABridge


def test_encoder_shape_and_determinism():
    enc = JEPAEncoder(HarnessConfig(jepa_embed_dim=64))
    a = enc.encode("hello")
    b = enc.encode("hello")
    assert a.shape == (1, 64)
    assert np.allclose(a, b)  # deterministic


def test_encoder_distinguishes_inputs():
    enc = JEPAEncoder(HarnessConfig(jepa_embed_dim=64))
    assert enc.similarity("running", "running") > 0.999
    assert enc.similarity("running", "error") < 0.9


def test_batch_encoding():
    enc = JEPAEncoder(HarnessConfig(jepa_embed_dim=32))
    out = enc.encode(["a", "b", "c"])
    assert out.shape == (3, 32)


def test_bridge_describe_and_references():
    bridge = JEPABridge(HarnessConfig(jepa_embed_dim=64))
    bridge.register_reference("running", {"status": "running"})
    desc = bridge.describe({"status": "running"})
    assert desc.nearest_reference == "running"
    assert desc.nearest_score > 0.99
    block = desc.as_prompt_block()
    assert "[PERCEIVED STATE]" in block


def test_bridge_projection_to_llm_hidden():
    bridge = JEPABridge(HarnessConfig(jepa_embed_dim=64), llm_hidden_size=16)
    projected = bridge.project("x")
    assert projected.shape == (16,)
