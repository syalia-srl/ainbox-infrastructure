import pytest
from ainbox_gateway.pool import Backend, Pool
from ainbox_gateway.router import Router, UnknownModel


def _router():
    pools = {
        "a": Pool("a", [Backend("a", "http://h:9000"), Backend("a", "http://h:9001")]),
        "b": Pool("b", [Backend("b", "http://h:9002")]),
    }
    return Router(pools)


def test_resolve_round_robins_within_slug():
    r = _router()
    urls = [r.resolve("a").base_url for _ in range(3)]
    assert urls == ["http://h:9000", "http://h:9001", "http://h:9000"]


def test_resolve_unknown_model_raises():
    with pytest.raises(UnknownModel):
        _router().resolve("nope")


def test_models_lists_sorted_slugs():
    assert _router().models() == ["a", "b"]
