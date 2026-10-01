"""Driver: truncate COCOImageTextDataset to N items, then run an entry point.
usage: python run_small_coco.py N main_mmae|test_mmae_updated [hydra overrides...]"""
import sys, os
sys.path.insert(0, os.getcwd())
import src.dataset.coco_dataset as cd

N = int(sys.argv[1]); target = sys.argv[2]; rest = sys.argv[3:]
_orig = cd.COCOImageTextDataset.__init__
def _init(self, *a, **k):
    _orig(self, *a, **k)
    self.data = self.data[:N]
cd.COCOImageTextDataset.__init__ = _init

if target == "main_mmae":
    sys.argv = ["main_mmae.py"] + rest
    import runpy
    runpy.run_path('main_mmae.py', run_name='__main__')
else:
    sys.path.insert(0, "src/test")
    import test_mmae_updated as t
    ok = t.test_mmae_training()
    t.test_wandb_logging()
    print("TEST_RESULT", ok)
