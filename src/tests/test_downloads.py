from types import SimpleNamespace

import pytest

from app import downloads


@pytest.fixture
def pull(monkeypatch, tmp_path):
    files = [("config.json", 10), ("model.safetensors", 100), ("pytorch_model.bin", 100)]
    monkeypatch.setattr(downloads.hf, "repo_files", lambda repo: files)
    monkeypatch.setattr(downloads, "try_to_load_from_cache", lambda *args: None)
    monkeypatch.setattr(downloads, "_jobs", {})

    def finish():
        for name, size in files:
            if name != "pytorch_model.bin":
                (tmp_path / name).write_bytes(b"x" * size)
        return str(tmp_path)

    return downloads.Job("org/model"), tmp_path, finish


def test_counts_reconstruction_not_network_transfer(pull, monkeypatch):
    job, _, finish = pull

    def snapshot(repo, allow_patterns, tqdm_class):
        assert allow_patterns == ["config.json", "model.safetensors"]
        assert job.total == 110
        with tqdm_class(unit="B", name="huggingface_hub.snapshot_download.transfer", total=110) as transfer:
            with tqdm_class(unit="B", name="huggingface_hub.snapshot_download", total=110) as reconstructed:
                transfer.update(110)
                reconstructed.update(55)
                assert job.done == 55
                assert job.view()["percent"] == 50
                assert not job.view()["finalizing"]
                reconstructed.update(55)
                assert job.view()["finalizing"]
                assert job.view()["percent"] < 100
                assert job.status == "downloading"
        return finish()

    monkeypatch.setattr(downloads, "snapshot_download", snapshot)
    downloads._run(job)
    assert job.status == "done"
    assert job.view()["percent"] == 100
    assert not job.view()["finalizing"]
    assert job.finished is not None


def test_cached_and_resumed_bytes_are_counted_once(pull, monkeypatch):
    job, cache, finish = pull
    cached = cache / "config.json"
    cached.write_bytes(b"x" * 10)
    monkeypatch.setattr(downloads, "try_to_load_from_cache",
                        lambda repo, name: str(cached) if name == "config.json" else None)

    def snapshot(repo, allow_patterns, tqdm_class):
        assert job.done == 10
        with tqdm_class(unit="files", total=2) as files:
            files.update(1)
            assert job.done == 10
        with tqdm_class(unit="B", total=100, initial=40) as bar:
            assert job.done == 50
            bar.update(60)
            assert job.done == 110
        return finish()

    monkeypatch.setattr(downloads, "snapshot_download", snapshot)
    downloads._run(job)
    assert job.status == "done"


@pytest.mark.parametrize("cached_size", [0, 5, None])
def test_partial_cache_is_not_counted_as_complete(pull, monkeypatch, cached_size):
    job, cache, _ = pull
    cached = cache / "config.json"
    if cached_size is not None:
        cached.write_bytes(b"x" * cached_size)
    monkeypatch.setattr(downloads, "try_to_load_from_cache", lambda *args: str(cached))
    assert downloads._cached_bytes(job.repo_id, [("config.json", 10)]) == 0


@pytest.mark.parametrize("mode", ["missing", "truncated"])
def test_snapshot_return_does_not_mean_selected_files_exist(pull, monkeypatch, mode):
    job, cache, finish = pull

    def snapshot(*args, **kwargs):
        finish()
        if mode == "missing":
            (cache / "model.safetensors").unlink()
        else:
            (cache / "model.safetensors").write_bytes(b"x")
        return str(cache)

    monkeypatch.setattr(downloads, "snapshot_download", snapshot)
    downloads._run(job)
    assert job.status == "error"
    assert "Snapshot file is missing or incomplete: model.safetensors" in job.error
    assert job.view()["percent"] < 100


def test_cancel_during_finalization_never_marks_done(pull, monkeypatch):
    job, _, finish = pull
    downloads._jobs[job.id] = job

    def snapshot(*args, **kwargs):
        job.add(job.total)
        assert job.view()["finalizing"]
        assert downloads.cancel(job.id)
        return finish()

    monkeypatch.setattr(downloads, "snapshot_download", snapshot)
    downloads._run(job)
    assert job.status == "cancelled"
    assert not job.view()["finalizing"]


def test_cancel_is_checked_on_network_transfer_updates(pull, monkeypatch):
    job, _, _ = pull

    def snapshot(repo, allow_patterns, tqdm_class):
        with tqdm_class(unit="B", name="huggingface_hub.snapshot_download.transfer", total=110) as bar:
            job.cancel.set()
            bar.update(1)
        pytest.fail("Cancelled transfer must abort")

    monkeypatch.setattr(downloads, "snapshot_download", snapshot)
    downloads._run(job)
    assert job.status == "cancelled"


def test_empty_selection_cannot_be_reported_as_done(pull, monkeypatch):
    job, _, _ = pull
    monkeypatch.setattr(downloads.hf, "repo_files", lambda repo: [("model.gguf", 100)])
    monkeypatch.setattr(downloads, "snapshot_download", lambda *args, **kw: pytest.fail("No selected files"))
    downloads._run(job)
    assert job.status == "error"
    assert job.error == "RuntimeError: No downloadable model files found"


def test_failure_after_full_byte_progress_is_not_completion(pull, monkeypatch):
    job, _, _ = pull

    def snapshot(*args, **kwargs):
        job.add(job.total)
        raise OSError("disk write failed")

    monkeypatch.setattr(downloads, "snapshot_download", snapshot)
    downloads._run(job)
    assert job.status == "error"
    assert job.error == "OSError: disk write failed"
    assert job.view()["percent"] < 100


def test_progress_retry_can_subtract_resumed_bytes():
    job = downloads.Job("org/model")
    job.total = 100
    job.add(40)
    job.add(-40)
    job.add(100)
    assert job.done == 100
    assert job.view()["speed_bps"] >= 0


def test_delete_cache_removes_only_its_completed_jobs(monkeypatch):
    done = downloads.Job("org/model")
    done.status = "done"
    other = downloads.Job("org/other")
    other.status = "done"
    failed = downloads.Job("org/model")
    failed.status = "error"
    jobs = {j.id: j for j in (done, other, failed)}
    monkeypatch.setattr(downloads, "_jobs", jobs)
    revisions = []
    monkeypatch.setattr(downloads, "scan_cache_dir", lambda: SimpleNamespace(
        repos=[SimpleNamespace(repo_id=done.repo_id, repo_type="model",
                               revisions=[SimpleNamespace(commit_hash="revision")])],
        delete_revisions=lambda *hashes: SimpleNamespace(execute=lambda: revisions.extend(hashes)),
    ))
    assert downloads.delete_cached(done.repo_id)
    assert revisions == ["revision"]
    assert set(jobs) == {other.id, failed.id}


def test_real_snapshot_progress_adapter(pull, monkeypatch):
    """Exercise installed HF aggregation, with only metadata and file I/O faked."""
    import huggingface_hub._snapshot_download as snapshots

    if not hasattr(snapshots, "read_tree_cache"):
        pytest.skip("Older HF uses per-file bars, covered by the resume test")
    job, cache, _ = pull
    revision = "a" * 40
    increments = []
    add = job.add

    def record(n):
        increments.append(n)
        add(n)

    monkeypatch.setattr(job, "add", record)
    monkeypatch.setattr(snapshots, "HfApi", lambda **kw: SimpleNamespace(
        repo_info=lambda **kw: SimpleNamespace(sha=revision),
    ))
    monkeypatch.setattr(snapshots, "read_tree_cache", lambda *args: {
        "config.json": SimpleNamespace(size=10),
        "model.safetensors": SimpleNamespace(size=100),
        "pytorch_model.bin": SimpleNamespace(size=100),
    })
    snapshot_path = cache / "models--org--model" / "snapshots" / revision

    def download(repo, filename, tqdm_class, **kwargs):
        size = 10 if filename == "config.json" else 100
        with tqdm_class(total=size, initial=0, unit="B") as progress:
            progress.update(size)
            if callable(transfer := getattr(progress, "update_transfer", None)):
                transfer(size)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        path = snapshot_path / filename
        path.write_bytes(b"x" * size)
        return str(path)

    monkeypatch.setattr(snapshots, "hf_hub_download", download)
    monkeypatch.setattr(downloads, "snapshot_download", lambda *args, **kwargs: snapshots.snapshot_download(
        *args, cache_dir=cache, **kwargs,
    ))
    downloads._run(job)
    assert job.status == "done", job.error
    assert job.done == job.total == 110
    assert sum(increments) == 110
