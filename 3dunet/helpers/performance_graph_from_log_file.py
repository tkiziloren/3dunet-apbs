import argparse
import re
import matplotlib.pyplot as plt


def parse_log_file(log_file_path):
    training_losses = []
    training_f1_scores = []
    validation_losses = []
    validation_f1_scores = []

    train_pattern = re.compile(r"Train Loss:\s*([\d\.]+),\s*F1:\s*([\d\.]+)")
    val_pattern = re.compile(r"Validation Loss:\s*([\d\.]+),\s*F1:\s*([\d\.]+)")

    with open(log_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        train_match = train_pattern.search(line)
        val_match = val_pattern.search(line)

        if train_match:
            t_loss = float(train_match.group(1))
            t_f1 = float(train_match.group(2))
            training_losses.append(t_loss)
            training_f1_scores.append(t_f1)

        if val_match:
            v_loss = float(val_match.group(1))
            v_f1 = float(val_match.group(2))
            validation_losses.append(v_loss)
            validation_f1_scores.append(v_f1)

    return training_losses, training_f1_scores, validation_losses, validation_f1_scores


def plot_metrics(training_losses, training_f1_scores, validation_losses, validation_f1_scores):
    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(14, 6))

    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss', marker='o')
    plt.plot(epochs, validation_losses, label='Validation Loss', marker='o')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Minimum değerleri yazdırma (grafiğin ortasına)
    min_train_loss = min(training_losses)
    min_val_loss = min(validation_losses)
    plt.text(len(epochs) / 2, (max(training_losses) + min(training_losses)) / 2,
             f'Min Training Loss: {min_train_loss:.4f}\nMin Validation Loss: {min_val_loss:.4f}',
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

    # F1 grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_f1_scores, label='Training F1', marker='o')
    plt.plot(epochs, validation_f1_scores, label='Validation F1', marker='o')
    plt.title('Training & Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # Maksimum değerleri yazdırma (grafiğin ortasına)
    max_train_f1 = max(training_f1_scores)
    max_val_f1 = max(validation_f1_scores)
    plt.text(len(epochs) / 2, (max(training_f1_scores) + min(training_f1_scores)) / 2,
             f'Max Training  F1: {max_train_f1:.4f}\nMax Validation F1: {max_val_f1:.4f}',
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot training and validation metrics from a log file.')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    args = parser.parse_args()

    training_losses, training_f1_scores, validation_losses, validation_f1_scores = parse_log_file(args.log_file)
    plot_metrics(training_losses, training_f1_scores, validation_losses, validation_f1_scores)


if __name__ == "__main__":
    main()