"""Evaluate frozen saved bundles on development partitions without fitting anything."""
import argparse
import json
import joblib
from threadpoolctl import threadpool_limits
from common import FILES, ROOT, development, features, new_output
from train_primary import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", choices=("validation", "train", "both"), default="validation")
    args = parser.parse_args()
    rows, fit, held, _ = development()
    partitions = [("train", fit), ("validation", held)]
    if args.partition != "both":
        partitions = [(name, ids) for name, ids in partitions if name == args.partition]
    output = new_output(args.output)
    results = []
    for family, filename in FILES.items():
        bundle = joblib.load(ROOT / "app/models" / filename)
        for split, ids in partitions:
            selected = rows.iloc[ids]
            x = features(selected, bundle)
            predicted = bundle["pipeline"].predict(x)
            if bundle["mapping"] is not None:
                predicted = bundle["mapping"][predicted]
            result = {"model": family, **evaluate(selected.label.to_numpy(), predicted, split)}
            results.append(result)
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps([{k: row[k] for k in ("model", "split", "accuracy", "macro_f1")} for row in results], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
