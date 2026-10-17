"""Copy a measured, quality-validated candidate into an independent bundle."""

import argparse, json, shutil
from pathlib import Path
from embeddinggemma2_rocm._bundle import bundle_digest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--profile", required=True)
p.add_argument("--quality", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
profile = json.loads(Path(a.profile).read_text())
quality = json.loads(Path(a.quality).read_text())
if (
    not profile.get("selected")
    or not quality.get("passed")
    or quality.get("images") != 256
    or quality.get("captions") != 1280
):
    raise ValueError(
        "A selected candidate and a passing complete quality report are required"
    )
source = Path(profile["selected"]["bundle"]).resolve()
destination = Path(a.output).resolve()
if source == destination or destination.exists():
    raise ValueError("Use a new output directory")
# The quality report must identify precisely this candidate, including its profiles.
index = json.loads((source / "index.json").read_text())
tested = quality.get("capabilities", {})
if (
    Path(quality.get("bundle", "")).resolve() != source
    or tested.get("profiles") != index.get("profiles")
    or quality.get("budget") != profile.get("budget")
):
    raise ValueError("Quality report does not match candidate profiles/budget")
if quality.get("bundle_sha256") != bundle_digest(source):
    raise ValueError("Candidate runtime files changed after quality validation")
shutil.copytree(source, destination, symlinks=False)
(destination / "activation.json").write_text(
    json.dumps(
        dict(
            activated=True,
            bundle_sha256=quality["bundle_sha256"],
            profile=profile,
            quality=quality,
        ),
        indent=2,
    )
)
print(destination)
