"""(Re)build the comparison code trees in HARNESS_DIR (default out/work/).

baseline/            : `git archive 27f7e44` (pre-fix code), only created if missing; always fingerprint-checked
ablate_nofreeze/     : fixed src with the CLIP-encoder freeze (fix 2) removed
ablate_noprep_test/  : fixed src without `accelerator.prepare(retrieval_test_loader)` (fix 3d)
"""
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # repo root
# Work dir for the baseline and ablation trees; override with HARNESS_DIR.
S = os.environ.get("HARNESS_DIR") or os.path.join(HERE, "out", "work")
# Last commit before the 2026-10-01 fixes. Do NOT use HEAD: once the fixes are committed
# the "baseline" would silently become the fixed code and every before/after check would pass.
BASELINE_COMMIT = "27f7e44"

if not os.path.isdir(f"{S}/baseline/src"):
    os.makedirs(f"{S}/baseline", exist_ok=True)
    subprocess.run(
        f"git -C {R} archive {BASELINE_COMMIT} | tar -x -C {S}/baseline",
        shell=True,
        check=True,
    )

# Fingerprint: the pre-fix evalrank hard-codes `.cuda()`; the fix removed it.
_eval = f"{S}/baseline/src/hook/eval_fusionmmae.py"
if ".cuda()" not in open(_eval).read():
    raise SystemExit(
        f"baseline fingerprint failed: {_eval} has no `.cuda()`, so {S}/baseline is not the "
        f"pre-fix code (commit {BASELINE_COMMIT}). Delete {S}/baseline and rerun."
    )
print(f"baseline OK ({BASELINE_COMMIT}, .cuda() present in eval_fusionmmae.py)")

ignore = shutil.ignore_patterns("__pycache__", "test")


def tree(name, path, old, new, count):
    shutil.copytree(f"{R}/src", f"{S}/{name}/src", ignore=ignore, dirs_exist_ok=True)
    f = f"{S}/{name}/src/{path}"
    s = open(f).read()
    assert s.count(old) == count, (name, s.count(old))
    open(f, "w").write(s.replace(old, new))
    print("built", name)


tree(
    "ablate_nofreeze",
    "model/mmae.py",
    "        for param in self.vision_encoder.parameters():\n"
    "            param.requires_grad = False\n"
    "        for param in self.text_encoder.parameters():\n"
    "            param.requires_grad = False\n",
    "",
    2,
)
tree(
    "ablate_noprep_test",
    "hook/train_fusionmmae_multi_learner.py",
    "    retrieval_test_loader = accelerator.prepare(retrieval_test_loader)\n",
    "",
    1,
)
