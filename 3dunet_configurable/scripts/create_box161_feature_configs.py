import argparse
import random
from pathlib import Path

import yaml


FEATURE_SETS = {
    "electrostatic_shape": ["electrostatic_grid", "shape"],
    "electrostatic_shape_compact_chem": [
        "electrostatic_grid",
        "shape",
        "atomic_donor",
        "atomic_acceptor",
        "atomic_hydrophobic",
        "atomic_aromatic",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create local box161 configs for APBS feature comparison."
    )
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="box161_dataset")
    parser.add_argument("--total", type=int, default=93)
    parser.add_argument("--train-count", type=int, default=69)
    parser.add_argument("--validation-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def select_cases(h5_dir: Path, total: int, seed: int):
    cases = sorted(path.stem for path in h5_dir.glob("*.h5"))
    if len(cases) < total:
        raise SystemExit(f"Need {total} H5 files, found {len(cases)} in {h5_dir}")
    rng = random.Random(seed)
    selected = rng.sample(cases, total)
    return sorted(selected)


def main():
    args = parse_args()
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_count + args.validation_count != args.total:
        raise SystemExit("--train-count + --validation-count must equal --total")

    with open(args.base_config) as handle:
        base_config = yaml.safe_load(handle)

    selected_cases = select_cases(h5_dir, args.total, args.seed)
    split_cases = selected_cases[:]
    random.Random(args.seed).shuffle(split_cases)

    train_cases = split_cases[: args.train_count]
    validation_cases = split_cases[args.train_count :]

    split_path = output_dir / f"{args.prefix}_split.yml"
    with split_path.open("w") as handle:
        yaml.safe_dump(
            {
                "seed": args.seed,
                "h5_directory": str(h5_dir),
                "total_available": len(list(h5_dir.glob("*.h5"))),
                "total": args.total,
                "train_count": len(train_cases),
                "validation_count": len(validation_cases),
                "selected_cases_sorted": selected_cases,
                "train": train_cases,
                "validation": validation_cases,
            },
            handle,
            sort_keys=False,
        )

    written = []
    for suffix, features in FEATURE_SETS.items():
        config = dict(base_config)
        config["name"] = f"{args.prefix}_{suffix}"
        config["h5_directory"] = str(h5_dir) + "/"
        config["seed"] = args.seed
        config["features"] = features
        config["label"] = "binding_site_in_dataset"
        config["datasets"] = {
            "train": train_cases,
            "validation": validation_cases,
            "test": validation_cases[:1],
        }

        config["training"] = dict(config["training"])
        config["training"]["num_epochs"] = args.epochs
        config["training"]["batch_size"] = args.batch_size
        config["training"]["accumulation_steps"] = args.accumulation_steps

        config["validation"] = dict(config["validation"])
        config["validation"]["batch_size"] = args.batch_size

        config_path = output_dir / f"{config['name']}.yml"
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        written.append(config_path)

    print(f"Split: {split_path}")
    print("Configs:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
