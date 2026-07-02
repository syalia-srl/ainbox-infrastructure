import pytest
from ainbox_gateway.pool import Backend, Pool


def test_single_backend_always_returned():
    b = Backend(slug="m", base_url="http://127.0.0.1:9000")
    pool = Pool(slug="m", backends=[b])
    assert [pool.next() for _ in range(3)] == [b, b, b]


def test_round_robin_cycles_in_order():
    bs = [Backend("m", f"http://127.0.0.1:{p}") for p in (9000, 9001, 9002)]
    pool = Pool(slug="m", backends=bs)
    got = [pool.next() for _ in range(7)]
    assert got == [bs[0], bs[1], bs[2], bs[0], bs[1], bs[2], bs[0]]


def test_empty_pool_rejected():
    with pytest.raises(ValueError):
        Pool(slug="m", backends=[])
