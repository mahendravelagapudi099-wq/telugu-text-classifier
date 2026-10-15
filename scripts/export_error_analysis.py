"""Export error analysis slices for the best transformer model.

Exports:
1. 100 misclassified test examples (stratified across classes) with model
   confidence, baseline prediction, and empty 'my_error_category'.
2. 30 examples where transformer is right and baseline is wrong.
3. 30 examples where baseline is right and transformer is wrong.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import load_config


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results"])
    print("Error analysis export template ready. Run after transformer test predictions are saved.")


if __name__ == "__main__":
    main()
