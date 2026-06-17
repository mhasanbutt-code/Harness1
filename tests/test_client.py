from harness.client import HarnessClient


def test_base_url_is_normalized():
    assert HarnessClient("http://host:8000/").base_url == "http://host:8000"
    assert HarnessClient("http://host:8000").base_url == "http://host:8000"


def test_default_points_at_localhost():
    assert HarnessClient().base_url == "http://127.0.0.1:8000"


def test_request_methods_exist():
    hc = HarnessClient()
    for name in ("health", "perceive", "chat", "ask", "ingest", "eval", "route"):
        assert callable(getattr(hc, name))
