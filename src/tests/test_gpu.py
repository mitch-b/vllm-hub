from app import gpu


def test_parse_discrete_gpu():
    info = gpu._parse("NVIDIA GeForce RTX 4090, 24564, 1200, 560.94\n")
    assert info["total_mib"] == 24564 and info["used_mib"] == 1200
    assert info["unified_memory"] is False


def test_parse_unified_memory_gpu_uses_system_ram(monkeypatch):
    monkeypatch.setattr(gpu, "_system_memory_mib", lambda: 124000)
    info = gpu._parse("NVIDIA GB10, [N/A], [N/A], 580.95\n")
    assert info["name"] == "NVIDIA GB10"
    assert info["total_mib"] == 124000 and info["used_mib"] == 0
    assert info["unified_memory"] is True


def test_nvidia_ngc_image_gets_vllm_serve(monkeypatch):
    from app import runner

    args = ["--model", "org/m", "--max-model-len", "8192"]
    assert runner._command(args) == args
    monkeypatch.setattr(runner, "IMAGE", "nvcr.io/nvidia/vllm:26.02-py3")
    assert runner._command(args) == ["vllm", "serve", "org/m", "--max-model-len", "8192"]


def test_library_hides_config_only_cache_entries(monkeypatch):
    from types import SimpleNamespace as NS

    from app import downloads

    def repo(name, files):
        rev = NS(files=[NS(file_name=f) for f in files], commit_hash="h")
        return NS(repo_id=name, repo_type="model", size_on_disk=1, last_accessed=0, revisions=[rev])

    repos = [repo("org/stub", ["config.json"]), repo("org/real", ["config.json", "model.safetensors"])]
    monkeypatch.setattr(downloads, "scan_cache_dir", lambda: NS(repos=repos))
    assert [m["id"] for m in downloads.cached_models()] == ["org/real"]
