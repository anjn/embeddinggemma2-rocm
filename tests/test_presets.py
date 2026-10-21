import copy
import json

import pytest
from embeddinggemma2_rocm import presets


def recipe():
    return dict(
        schema=1,
        kind="hip_tuning_recipe",
        profile=dict(batch=4),
        vision_batch=4,
        kernel_config=dict(segmented=False, qk=dict(block=128, lanes=16, packed=True)),
        environment=dict(
            device=dict(arch="gfx1150", wave_size=32),
            limits=dict(max_threads_per_block=1024),
            software=dict(rocm="test"),
            recipe_sha256="test",
        ),
        validation=dict(
            numeric_pass=True,
            retrieval_pass=True,
            min_cosine_to_standard=0.999,
            retrieval=dict(text_to_image=80.0, image_to_text=90.0),
            loss_percentage_points=dict(text_to_image=0.0, image_to_text=0.0),
        ),
    )


def test_matching_recipe_requires_exact_environment(tmp_path):
    r = recipe()
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(r))
    assert presets.select(r["profile"], r["environment"], path) == r
    env = copy.deepcopy(r["environment"])
    env["software"]["rocm"] = "other"
    with pytest.raises(ValueError, match="software"):
        presets.select(r["profile"], env, path)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(vision_batch=3),
        lambda r: r["kernel_config"]["qk"].update(lanes=64, packed=False),
        lambda r: r["validation"].update(retrieval_pass=False),
        lambda r: r["validation"].update(min_cosine_to_standard=float("nan")),
        lambda r: r["kernel_config"].update(segmented="false"),
    ],
)
def test_invalid_recipe_rejected(change):
    r = recipe()
    change(r)
    with pytest.raises(ValueError):
        presets.validate(r)


def test_missing_preset_does_not_tune(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "registry", lambda: tmp_path)
    r = recipe()
    assert presets.select(r["profile"], r["environment"]) is None


def test_default_path_can_win_search():
    r = recipe()
    r["kernel_config"] = None
    assert presets.validate(r) == r


def test_export_matches_settings_and_emits_no_local_paths(tmp_path, monkeypatch):
    from embeddinggemma2_rocm import _bundle
    from embeddinggemma2_rocm.preparation import write_json

    bundle = tmp_path / "private-bundle"
    key = dict(
        model_revision="revision", budget=70, batch=4, aspect=[16, 9], precision="fast"
    )
    r = recipe()
    r["profile"] = key
    write_json(
        bundle / "index.json",
        dict(
            revision="revision",
            profiles=[
                dict(
                    path="image",
                    **{k: v for k, v in key.items() if k != "model_revision"},
                )
            ],
        ),
    )
    write_json(
        bundle / "image/bundle.json",
        dict(vision=dict(batch=4), fingerprint=dict(kernel_config=r["kernel_config"])),
    )
    profile = tmp_path / "profile.json"
    quality = tmp_path / "quality.json"
    write_json(
        profile,
        dict(
            complete=True,
            budget=70,
            environment=r["environment"],
            profile=key,
            selected=dict(
                bundle=str(bundle),
                vision_batch=4,
                kernel_config=r["kernel_config"],
                ms_per_image=10.0,
                min_cosine_to_standard=0.999,
            ),
        ),
    )
    write_json(
        quality,
        dict(
            passed=True,
            images=256,
            captions=1280,
            bundle=str(bundle),
            budget=70,
            bundle_sha256="digest",
            candidate=r["validation"]["retrieval"],
            reference=r["validation"]["retrieval"],
            loss_percentage_points=r["validation"]["loss_percentage_points"],
            input_sha256="inputs",
            manifest_sha256="manifest",
        ),
    )
    monkeypatch.setattr(_bundle, "bundle_digest", lambda _: "digest")
    destination = tmp_path / "portable.json"
    presets.export_recipe(profile, quality, destination)
    assert str(tmp_path) not in destination.read_text()
    assert json.loads(destination.read_text())["kernel_config"] == r["kernel_config"]
    changed = json.loads(profile.read_text())
    changed["selected"]["vision_batch"] = 1
    write_json(profile, changed)
    with pytest.raises(ValueError, match="settings differ"):
        presets.export_recipe(profile, quality, tmp_path / "bad.json")


def test_published_recipes_match_shipped_compiler_sources():
    for path in presets.registry().rglob("*.json"):
        record = presets.validate(json.loads(path.read_text()))
        assert record["environment"]["recipe_sha256"] == presets.recipe_digest(), path
        assert record["validation"]["images"] == 256
        assert record["validation"]["captions"] == 1280
