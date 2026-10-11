"""Downloaded language selection must agree with the pinned release catalog."""

import copy
import json

import huggingface_hub
import pytest
import torch

from kokoro_multispeaker import bundles


def fixed_bundle(tmp_path, damage=None):
    names = ["AV_ariala", "AV_brody"]
    conditioning = {
        "format": "materialized-fixed-speakers-v1",
        "all510_rows_identical_per_voice": True,
        "decoder_and_predictor_halves_finalized": True,
        "voicepacks": {
            name: {
                "file": "voices/" + name + ".pt",
                "sha256": "d" * 64,
                "shape": [510, 1, 256],
            }
            for name in names
        },
    }
    config = {
        "format": "akinvox-stock-language-bundle/v1",
        "language": "en",
        "base_sha256": bundles.BASE_SHA256,
        "speaker_names": names,
        "update": 185000,
        "source_checkpoint_sha256": "c" * 64,
        "voice_conditioning": copy.deepcopy(conditioning),
    }
    bank = torch.stack([torch.ones(510, 256), torch.full((510, 256), 2.0)])
    release = {
        "contract": "akinvox-stock-multispeaker-adapter/v1",
        "task": "en_multispeaker",
        "base_sha256": bundles.BASE_SHA256,
        "update": 185000,
        "source_checkpoint_sha256": "c" * 64,
        "speaker_names": names,
        "voicebank": bank,
        "targets": [{} for _ in range(201)],
        "state": {str(i): torch.ones(1) for i in range(402)},
        "voice_conditioning": conditioning,
    }
    if damage == "checkpoint":
        config["source_checkpoint_sha256"] = "e" * 64
    elif damage == "update":
        config["update"] += 1
    elif damage == "profile":
        config["voice_conditioning"]["voicepacks"][names[0]]["sha256"] = "e" * 64
    elif damage == "half":
        release["voice_conditioning"]["decoder_and_predictor_halves_finalized"] = False
        config["voice_conditioning"] = copy.deepcopy(release["voice_conditioning"])
    elif damage == "name":
        release["voice_conditioning"]["voicepacks"]["AV_wrong"] = release[
            "voice_conditioning"
        ]["voicepacks"].pop(names[0])
        config["voice_conditioning"] = copy.deepcopy(release["voice_conditioning"])
    elif damage == "raw_row":
        release["voicebank"][0, 8] += 1
    torch.save(release, tmp_path / "adapter.pt")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            bundles.sha(tmp_path / name) + "  " + name + "\n"
            for name in ["adapter.pt", "config.json"]
        )
    )
    return tmp_path


def test_fixed_bundle_retains_bound_native_profiles(tmp_path):
    release, config = bundles._checked_bundle(fixed_bundle(tmp_path), "en")
    assert release["source_checkpoint_sha256"] == config["source_checkpoint_sha256"]
    assert torch.equal(
        release["voicebank"],
        release["voicebank"][:, 0:1].expand_as(release["voicebank"]),
    )


@pytest.mark.parametrize(
    "damage", ["checkpoint", "update", "profile", "half", "name", "raw_row"]
)
def test_recomputed_checksums_do_not_hide_wrong_profile_or_checkpoint(tmp_path, damage):
    with pytest.raises(ValueError, match="profiles|voicebank"):
        bundles._checked_bundle(fixed_bundle(tmp_path, damage), "en")


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    registry = {}
    for language in ("en", "de"):
        folder = tmp_path / "adapters" / language
        folder.mkdir(parents=True)
        (folder / "adapter.pt").write_bytes(language.encode())
        (folder / "config.json").write_text("{}")
        registry[language] = {
            "repo": "AKinvox/kokoro-multispeaker",
            "revision": "pinned-revision",
            "path": "adapters/" + language,
            "adapter_sha256": bundles.sha(folder / "adapter.pt"),
            "config_sha256": bundles.sha(folder / "config.json"),
            "speaker_names": ["AV_example"],
        }
    (tmp_path / "releases.json").write_text(json.dumps(registry))
    catalog = {
        "format": "akinvox-multispeaker-catalog/v1",
        "base_sha256": bundles.BASE_SHA256,
        "languages": {
            language: {
                key: value
                for key, value in item.items()
                if key not in {"repo", "revision"}
            }
            for language, item in registry.items()
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(catalog))
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(bundles, "__file__", str(tmp_path / "bundles.py"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return tmp_path, catalog, calls


@pytest.mark.parametrize("language", ["en", "de"])
def test_catalog_is_retrieved_with_only_the_selected_adapter(snapshot, language):
    root, _, calls = snapshot
    assert bundles._default_bundle(language) == root / "adapters" / language
    repo, kwargs = calls[0]
    assert repo == "AKinvox/kokoro-multispeaker"
    assert kwargs["revision"] == "pinned-revision"
    assert set(kwargs["allow_patterns"]) == {
        "config.json",
        "adapters/" + language + "/adapter.pt",
        "adapters/" + language + "/config.json",
        "adapters/" + language + "/SHA256SUMS",
    }


@pytest.mark.parametrize("field", ["base_sha256", "path", "speaker_names"])
def test_catalog_disagreement_fails_before_adapter_use(snapshot, field):
    root, catalog, _ = snapshot
    if field == "base_sha256":
        catalog[field] = "wrong-base"
    else:
        catalog["languages"]["en"][field] = "wrong-entry"
    (root / "config.json").write_text(json.dumps(catalog))
    (root / "adapters/en/adapter.pt").unlink()
    with pytest.raises(ValueError, match="catalog differs"):
        bundles._default_bundle("en")


def test_missing_catalog_is_not_silently_ignored(snapshot):
    root, _, _ = snapshot
    (root / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        bundles._default_bundle("en")
