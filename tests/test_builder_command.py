from ainbox_builder.builder import build_command, Step


def test_build_command_no_push():
    steps = build_command("smaug_v1", "12.8.1-devel-ubuntu22.04", "registry.syalia.dev", push=False)
    assert steps == [
        Step(label="build",
             argv=["make", "image", "RECIPE=recipes/smaug_v1.json"],
             env={"CUDA_TAG": "12.8.1-devel-ubuntu22.04"}),
    ]


def test_build_command_with_push():
    steps = build_command("smaug_v1", "12.2.2-devel-ubuntu22.04", "registry.syalia.dev", push=True)
    labels = [s.label for s in steps]
    assert labels == ["build", "tag", "push"]
    assert steps[1].argv == ["docker", "tag", "superbot:smaug_v1",
                             "registry.syalia.dev/ainbox-infra/smaug_v1:latest"]
    assert steps[2].argv == ["docker", "push",
                             "registry.syalia.dev/ainbox-infra/smaug_v1:latest"]
