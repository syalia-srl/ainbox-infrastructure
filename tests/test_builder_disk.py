import pytest
from ainbox_builder.builder import parse_docker_root, disk_free_gb
from ainbox_builder.catalog import CATALOG, BASE_IMAGE_GB
from ainbox_builder.recipe import estimate_image_gb


def test_catalog_entries_carry_numeric_gb():
    for slug, meta in CATALOG["llm"].items():
        assert isinstance(meta["gb"], (int, float)) and meta["gb"] > 0


def test_base_image_gb_is_positive():
    assert BASE_IMAGE_GB > 0


def test_parse_docker_root_trims():
    assert parse_docker_root(b"/var/lib/docker\n") == "/var/lib/docker"
    assert parse_docker_root("/home/docker\n") == "/home/docker"


def test_disk_free_gb_real_path(tmp_path):
    d = disk_free_gb(str(tmp_path))
    assert d["free_gb"] > 0 and d["total_gb"] >= d["free_gb"]
    assert d["used_gb"] == pytest.approx(d["total_gb"] - d["free_gb"], abs=0.5)


def test_estimate_sums_base_plus_selected():
    sel = {"llm": [{"alias": "qwen3-14b"}], "stt": [{"alias": "whisper-small"}]}
    est = estimate_image_gb(sel, CATALOG, BASE_IMAGE_GB)
    expected = BASE_IMAGE_GB + CATALOG["llm"]["qwen3-14b"]["gb"] + CATALOG["stt"]["whisper-small"]["gb"]
    assert est == pytest.approx(expected)


def test_estimate_custom_url_counts_as_unknown():
    sel = {"llm": [{"alias": "mystery", "url": "https://hf/x.gguf"}]}
    est = estimate_image_gb(sel, CATALOG, BASE_IMAGE_GB)
    # unknown model contributes a default guess, so estimate exceeds base alone
    assert est > BASE_IMAGE_GB
