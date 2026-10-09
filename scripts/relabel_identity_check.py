"""Identity check before evaluation: arm R checkpoints must have been trained on skin31_relabel, i.e. predict the
64 relabelled training images (SK folder -> Melanoma) as melanoma far more often than arm A does."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import milk_eval as M  # noqa: E402
import pad_eval as P  # noqa: E402

M.ARMS["R"] = ROOT / "checkpoints_relabel"
names = np.array(P.class_names("clean"))
man = pd.read_csv(ROOT / "data/skin31_relabel/manifest.csv")
paths = [ROOT / "data/skin31_relabel" / d for d in man[man.reason.str.startswith("relabel SK", na=False)].dst]
print(len(paths), "relabelled training images")
for arm in ("A", "R"):
    acc = [(names[P.logits(M.load(arm, s), paths, 30).argmax(1)] == "melanoma").mean() for s in M.SEEDS]
    print(arm, "predicted melanoma:", np.round(acc, 3))
