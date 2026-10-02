"""The launch scripts hand train.py overrides that Hydra accepts and that carry the note verbatim."""
import os
import subprocess

import pytest
from omegaconf import OmegaConf

from helpers import REPO, compose_cfg

FAKE_ACCELERATE = """#!/usr/bin/env bash
printf '%s\\0' "$@" > "$FAKE_OUT/args"
printf '%s' "${MMAE_NOTE-<unset>}" > "$FAKE_OUT/note"
"""
NOTES = ["Wangyuan's try: a=1, b", 'say "hi" \\ ${oc.env:HOME} [x], y=z']


def launch(tmp_path, script: str, *args: str) -> tuple[list[str], str]:
    """Run scripts/<script> with a stub `accelerate` on PATH; return the arguments it gives train.py and
    the MMAE_NOTE it exports."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "accelerate"
    stub.write_text(FAKE_ACCELERATE)
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_OUT": str(tmp_path), "CUDA_VISIBLE_DEVICES": "0"}
    env.pop("MMAE_NOTE", None)
    result = subprocess.run(["bash", str(REPO / "scripts" / script), *args], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "args").read_text().split("\0")[:-1]
    return argv[argv.index("train.py") + 1 :], (tmp_path / "note").read_text()


@pytest.mark.parametrize("note", NOTES)
@pytest.mark.parametrize("script", ["run_local.sh", "run_cluster.sh"])
def test_launch_scripts_pass_any_note_verbatim(tmp_path, monkeypatch, script, note):
    args = (note, "train=debug") if script == "run_local.sh" else ("64", "3", note, "train=debug")
    overrides, env_note = launch(tmp_path, script, *args)
    monkeypatch.setenv("MMAE_NOTE", env_note)
    cfg = OmegaConf.to_container(compose_cfg(*overrides), resolve=True)  # what train.py's config.yaml holds
    assert cfg["wandb"]["notes"] == note
    assert cfg["wandb"]["tags"] == (["local"] if script == "run_local.sh" else ["cluster"])


def test_run_eval_calls_evaluate_with_cluster_data_and_forwards_overrides(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text("#!/usr/bin/env bash\nprintf '%s\\0' \"$@\" > \"$FAKE_OUT/args\"\n")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_OUT": str(tmp_path)}
    overrides = ["eval.run_dir=/a/b", "eval.output=/x y/z.json", "wandb.notes=${oc.env:HOME}"]
    result = subprocess.run(["bash", str(REPO / "scripts" / "run_eval.sh"), *overrides], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "args").read_text().split("\0")[:-1]
    assert argv == ["evaluate.py", "data=coco_cluster", *overrides]


def test_run_eval_script_text_avoids_cluster_tool_triggers():
    text = (REPO / "scripts" / "run_eval.sh").read_text()
    assert "CLUSTER_DRY_RUN" not in text
    assert "CUDA_VISIBLE_DEVICES" not in text
