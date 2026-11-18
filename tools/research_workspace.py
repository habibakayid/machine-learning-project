"""Copy archived research into a new workspace before rerunning historical studies."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "supplemental_field/research_project"
    target = args.output.resolve()
    if target == source or source in target.parents or target.exists():
        raise ValueError("Choose a new directory outside the archived research tree.")
    shutil.copytree(source, target)
    print(f"Research workspace: {target}")
    print("No training has been run. Restore the Kaggle data in that workspace if not imported with --research.")


if __name__ == "__main__":
    main()
