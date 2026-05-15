import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create extended box161 loss calibration configs for compact chemistry."
    )
    parser.add_argument(
        "--base-config",
        default="config/local/box161/box161_dataset_electrostatic_shape_compact_chem.yml",
    )
    parser.add_argument("--output-dir", default="config/local/box161_loss_calibration")
    parser.add_argument("--prefix", default="box161_compact")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--pos-weights", type=float, nargs="+", default=[5.0, 10.0, 25.0])
    return parser.parse_args()


def normalize_weight_name(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.base_config) as handle:
        base_config = yaml.safe_load(handle)

    written = []
    for pos_weight in args.pos_weights:
        config = dict(base_config)
        weight_name = normalize_weight_name(pos_weight)
        config["name"] = f"{args.prefix}_pos{weight_name}_long"

        config["training"] = dict(base_config["training"])
        config["training"]["num_epochs"] = args.epochs
        config["training"]["early_stopping_patience"] = args.early_stopping_patience

        scheduler = dict(config["training"].get("scheduler", {}))
        scheduler["patience"] = args.scheduler_patience
        config["training"]["scheduler"] = scheduler

        loss = dict(config["training"].get("loss", {}))
        loss["type"] = "BCEDiceLoss"
        loss["pos_weight"] = float(pos_weight)
        loss["dynamic_pos_weight"] = False
        config["training"]["loss"] = loss

        config_path = output_dir / f"{config['name']}.yml"
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        written.append(config_path)

    print("Configs:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
