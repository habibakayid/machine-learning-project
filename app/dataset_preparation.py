"""Release-local constants required by the unchanged feature modules."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "cache"
CLASSES = ("daisy", "dandelion", "rose", "sunflower", "tulip")
SEED = 42
