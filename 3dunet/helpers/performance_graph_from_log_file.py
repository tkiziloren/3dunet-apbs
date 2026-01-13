import argparse
import re
import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    training_losses = []
    training_f1_scores = []
    validation_losses = []
    validation_f1_scores = []

    # capture metrics
    train_pat = re.compile(r"Train Loss:\s*([\d\.]+),\s*F1:\s*([\d\.]+)")
    val_pat   = re.compile(r"Validation Loss:\s*([\d\.]+),\s*F1:\s*([\d\.]+)")

    # now the config block – in the order we expect them
    ordered_keys = [
        ("Configuration",   re.compile(r"Configuration name:\s*(\S+)")),
        ("Model",           re.compile(r"Model:\s*(\S+)")),
        ("Base features",   re.compile(r"Base features:\s*(\d+)")),
        ("Batch size",      re.compile(r"Batch size:\s*(\d+)")),
        ("Learning rate",   re.compile(r"Learning rate:\s*([\d\.eE-]+)")),
        ("Optimizer",       re.compile(r"Optimizer:\s*(\S+)")),
        ("Scheduler",       re.compile(r"Scheduler:\s*(\{.*\})")),
        ("Loss function",   re.compile(r"Loss function:\s*(\{.*\})")),
        ("Best val F1",     re.compile(r"Best validation F1 score:\s*([\d\.]+)")),
        ("Best train F1",   re.compile(r"best training f1 score:\s*([\d\.]+)")),
    ]
    cfg = {}

    with open(log_file_path) as f:
        for line in f:
            line = line.strip()
            # metrics
            if m := train_pat.search(line):
                training_losses.append(float(m.group(1)))
                training_f1_scores.append(float(m.group(2)))
            if m := val_pat.search(line):
                validation_losses.append(float(m.group(1)))
                validation_f1_scores.append(float(m.group(2)))

            # config kvs in order
            for key, patt in ordered_keys:
                if key not in cfg and (m := patt.search(line)):
                    cfg[key] = m.group(1)

    # we return the lists and the ordered cfg dict
    return training_losses, training_f1_scores, validation_losses, validation_f1_scores, cfg


def plot_metrics(tl, tf1, vl, vf1, cfg):
    epochs = range(1, len(tl) + 1)
    plt.figure(figsize=(14,6))

    # build config text block
    cfg_lines = [f"{k}: {cfg[k]}" for k, _ in [
        ("Configuration", None),
        ("Model", None),
        ("Base features", None),
        ("Batch size", None),
        ("Learning rate", None),
        ("Optimizer", None),
        ("Scheduler", None),
        ("Loss function", None),
        ("Best val F1", None),
        ("Best train F1", None),
    ] if k in cfg]
    print("------ Run Summary ------")
    print("Model:", cfg.get("Model", "Unknown"))
    print("Configuration:", cfg.get("Configuration", "Unknown"))
    print("Base features:", cfg.get("Base features", "Unknown"))
    print("Batch size:", cfg.get("Batch size", "Unknown"))
    print("Learning rate:", cfg.get("Learning rate", "Unknown"))
    print("Optimizer:", cfg.get("Optimizer", "Unknown"))
    print("Scheduler:", cfg.get("Scheduler", "Unknown"))
    print("Loss function:", cfg.get("Loss function", "Unknown"))
    print("Best val F1:", cfg.get("Best val F1", "Unknown"))
    print("Best train F1:", cfg.get("Best train F1", "Unknown"))
    cfg_text = "\n".join(cfg_lines)
    plt.suptitle(cfg_text, fontsize=10, y=0.95)

    # left: Loss
    ax1 = plt.subplot(1,2,1)
    ax1.plot(epochs, tl,   label="Training Loss",   marker="o")
    ax1.plot(epochs, vl,   label="Validation Loss", marker="o")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax1.text(0.5, 0.5,
             f"Min Train: {min(tl):.4f}\nMin Val: {min(vl):.4f}",
             transform=ax1.transAxes, ha="center", va="center",
             bbox=dict(facecolor="white", alpha=0.6))

    # right: F1
    ax2 = plt.subplot(1,2,2)
    ax2.plot(epochs, tf1,  label="Training F1",   marker="o")
    ax2.plot(epochs, vf1,  label="Validation F1", marker="o")
    ax2.set_title("F1 Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.legend()
    ax2.grid(True)
    ax2.text(0.5, 0.5,
             f"Max Train: {max(tf1):.4f}\nMax Val: {max(vf1):.4f}",
             transform=ax2.transAxes, ha="center", va="center",
             bbox=dict(facecolor="white", alpha=0.6))

    plt.tight_layout()
    plt.show()

FILE_LIST = [
    "../output/codon_new_structure/ConvNeXt3DV2_base_features64/ConvNeXt3DV2_base_features64_65709312.err",
    "../output/codon_new_structure/ConvNeXt3D_base_features64/ConvNeXt3D_base_features64_65708077.err",
    "../output/codon_new_structure/ResNet3D4L_base_features64/ResNet3D4L_base_features64_65692529.err",
    "../output/codon_new_structure/ResNet3D5L_base_features64/ResNet3D5L_base_features64_65693681.err",
    "../output/codon_new_structure/ResNet3D6L_base_features64/ResNet3D6L_base_features64_65693930.err",
    "../output/codon_new_structure/UNet3D4LA_base_features64/UNet3D4LA_base_features64_65682186.err",
    "../output/codon_new_structure/UNet3D4L_base_features64/UNet3D4L_base_features64_65679012.err",
    "../output/codon_new_structure/UNet3D5L_base_features64/UNet3D5L_base_features64_65681920.err",
    "../output/codon_new_structure/UNet3D6L_base_features64/UNet3D6L_base_features64_65682108.err",
    "../output/codon_new_structure/UNet3D4LA_pdbbind_optimized_pos_weight_1_batch_size_4/UNet3D4LA_pdbbind_optimized_pos_weight_1_batch_size_4_66093982.err",
    "../output/codon_new_structure/UNet3D4LA_pdbbind_randomized_config/UNet3D4LA_pdbbind_randomized_config_66241568.err",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=False, help="Path to your training log")
    args = parser.parse_args()

    if args.log_file:
        files = [args.log_file]
    else:
        # manuel tanımlı listeyi kullan
        files = FILE_LIST

    if not files:
        print("⚠️ İşlenecek dosya yok.")
        return

    for fpath in files:
        print(f"\n--- Processing {fpath} ---")
        tl, tf1, vl, vf1, meta = parse_log_file(fpath)
        plot_metrics(tl, tf1, vl, vf1, meta)


if __name__ == "__main__":
    main()