import ainbox_gateway


def test_package_has_version():
    assert isinstance(ainbox_gateway.__version__, str)
    assert ainbox_gateway.__version__
