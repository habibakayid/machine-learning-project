"""Run the actual original cleaning/grouping implementation into a new audit directory."""
import argparse
from pathlib import Path
from common import ROOT, new_output
import dataset_preparation as preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not preparation.RAW.is_dir():
        raise FileNotFoundError("Import the Kaggle images first.")
    output = new_output(args.output)
    preparation.RESULTS = output / "results"
    preparation.FIGURES = output / "figures"
    preparation.prepare_dataset()
    print("Audit outputs saved separately; frozen manifests were not replaced.")


if __name__ == "__main__":
    main()
