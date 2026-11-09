from pathlib import Path
from types import SimpleNamespace

import pytest
from docker.errors import ImageNotFound

from app import profiles, runner


def test_parser_uses_revision_pinned_shared_cache_path(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HF_HOME", str(tmp_path))
    calls = []
    relative = Path("hub/models--nvidia--model/snapshots/revision/super_v3_reasoning_parser.py")
    monkeypatch.setattr(runner, "hf_hub_download",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or str(tmp_path / relative))
    assert runner._spark_parser() == str(Path("/root/.cache/huggingface") / relative)
    assert calls == [((profiles.NEMOTRON_SUPER, "super_v3_reasoning_parser.py"),
                      {"revision": "ff433f5493e25d631c9f12b5d55c674229923d02"})]


def test_parser_outside_shared_cache_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "HF_HOME", str(tmp_path / "shared"))
    monkeypatch.setattr(runner, "hf_hub_download", lambda *a, **kw: str(tmp_path / "unshared.py"))
    with pytest.raises(ValueError):
        runner._spark_parser()


def test_recipe_image_pulled_only_when_missing(monkeypatch):
    calls = []
    def get(image):
        raise ImageNotFound("Not downloaded")
    images = SimpleNamespace(get=get, pull=lambda image: calls.append(image))
    monkeypatch.setattr(runner, "client", lambda: SimpleNamespace(images=images))
    runner._ensure_image(profiles.SPARK_IMAGE)
    assert calls == [profiles.SPARK_IMAGE]
    images.get = lambda image: object()
    runner._ensure_image(profiles.SPARK_IMAGE)
    assert calls == [profiles.SPARK_IMAGE]
