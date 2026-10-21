"""CPU checks for search coverage and interrupted worker recovery."""

import json

import pytest
from embeddinggemma2_rocm import tuning
from embeddinggemma2_rocm._compiler.search import candidates, combinations
from embeddinggemma2_rocm.preparation import write_json


def test_wave_size_and_block_limits():
    for wave in (32, 64):
        device = dict(wave_size=wave, max_threads_per_block=256)
        for family in ("gelu", "qk", "norm", "preprocess"):
            values = candidates(family, device)
            assert values and all(v["block"] <= 256 for v in values)
            assert all(v.get("lanes", 1) <= wave for v in values)
        assert any(v["lanes"] == wave for v in candidates("qk", device))


def test_all_pipeline_combinations():
    values = list(
        combinations(
            dict(
                gelu={"vector": 4},
                qk={"lanes": 16},
                norm={"lanes": 32},
                preprocess={"vector": 1},
            )
        )
    )
    assert len(values) == 32
    assert len({json.dumps(v, sort_keys=True) for v in values}) == 32
    assert any(
        not any(v[f] for f in ("gelu", "qk", "norm", "preprocess"))
        and not v["segmented"]
        for v in values
    )


def test_resume_does_not_retry_crashing_kernel(tmp_path, monkeypatch):
    output = tmp_path / "results.json"
    configs = [dict(block=64), dict(block=128)]
    write_json(output, [dict(config=configs[0], status="running")])

    def worker(job, path):
        records = json.loads(output.read_text())
        assert records[0]["status"] == "failed"
        records.append(
            dict(config=configs[1], status="ok", numeric_pass=True, median_seconds=0.1)
        )
        write_json(output, records)
        return 0

    monkeypatch.setattr(tuning, "_worker", worker)
    assert (
        tuning._screen(dict(output=str(output), candidates=configs), tmp_path)
        == configs[1]
    )


def test_no_progress_stops_instead_of_spinning(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "_worker", lambda *a: 0)
    with pytest.raises(RuntimeError, match="no progress"):
        tuning._screen(
            dict(output=str(tmp_path / "results.json"), candidates=[{}]), tmp_path
        )


def test_worker_timeout_is_checkpointable(tmp_path, monkeypatch):
    killed = []

    class Process:
        pid = 123

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, timeout=None):
            if timeout is not None:
                raise tuning.subprocess.TimeoutExpired("worker", timeout)
            return -9

    monkeypatch.setattr(tuning.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(tuning.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    assert tuning._worker(dict(kind="probe"), tmp_path / "job.json", 1) == 124
    assert killed == [(123, tuning.signal.SIGKILL)]


def test_empty_checkpoint_preserved_for_recovery(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"")
    assert tuning._read_checkpoint(path, []) == []
    assert not path.exists()
    assert len(list(tmp_path.glob("result.json.corrupt-*"))) == 1


def test_tuning_fixture_has_distinct_large_batch_images():
    import numpy as np
    from embeddinggemma2_rocm.benchmark import images

    legacy = images(32)
    distinct = images(32, unique=True)
    np.testing.assert_array_equal(legacy[:16], distinct[:16])
    np.testing.assert_array_equal(legacy[0], legacy[16])
    assert not np.array_equal(distinct[0], distinct[16])
