import pandas as pd
import matplotlib.pyplot as plt

# =======================
# GLOBAL CONFIG (EDIT THESE)
# =======================
# File paths
CSV_FILE_1 = "train_accuracy.csv"
CSV_FILE_2 = "train_loss.csv"
CSV_FILE_3 = "grad_norm.csv"

# Column names (edit if needed)
X_COL = "Step"
Y1_COL = "train_accuracy"
Y2_COL = "train_loss"
Y3_COL = "grad_norm"

# Titles and labels
TITLE_1 = "Training Accuracy"
TITLE_2 = "Training Loss"
TITLE_3 = "Gradient Norm"
X_LABEL = "Step"
Y1_LABEL = "Accuracy"
Y2_LABEL = "Loss"
Y3_LABEL = "Grad Norm"

# Font settings
FONT_FAMILY = "STIXGeneral"
TITLE_SIZE = 18
LABEL_SIZE = 14
TICK_SIZE = 12

# Line styling
LINE_WIDTH_1 = 1.2
LINE_WIDTH_2 = 1.2
LINE_WIDTH_3 = 1.2
LINE_COLOR_1 = "black"
LINE_COLOR_2 = "blue"
LINE_COLOR_3 = "crimson"
LINE_STYLE_1 = "-"
LINE_STYLE_2 = "-"
LINE_STYLE_3 = "-"

# Figure and subplot size
FIG_SIZE = (18, 5)   # widened for 3 subplots
DPI = 120

# =======================
# FUNCTION DEFINITIONS
# =======================
def style_axes(ax):
    """Remove top and right spines and apply clean style."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

def load_csv(file_path):
    return pd.read_csv(file_path)

# =======================
# MAIN PLOTTING
# =======================
def main():
    # Set font globally
    plt.rcParams['font.family'] = FONT_FAMILY

    # Load data
    df1 = load_csv(CSV_FILE_1)
    df2 = load_csv(CSV_FILE_2)
    df3 = load_csv(CSV_FILE_3)

    # Create subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, dpi=DPI)

    # ---- Plot 1 ----
    axes[0].plot(
        df1[X_COL], df1[Y1_COL],
        linewidth=LINE_WIDTH_1,
        color=LINE_COLOR_1,
        linestyle=LINE_STYLE_1
    )
    axes[0].set_title(TITLE_1, fontsize=TITLE_SIZE)
    axes[0].set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    axes[0].set_ylabel(Y1_LABEL, fontsize=LABEL_SIZE)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    style_axes(axes[0])

    # ---- Plot 2 ----
    axes[1].plot(
        df2[X_COL], df2[Y2_COL],
        linewidth=LINE_WIDTH_2,
        color=LINE_COLOR_2,
        linestyle=LINE_STYLE_2
    )
    axes[1].set_title(TITLE_2, fontsize=TITLE_SIZE)
    axes[1].set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    axes[1].set_ylabel(Y2_LABEL, fontsize=LABEL_SIZE)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    style_axes(axes[1])

    # ---- Plot 3 (grad_norm) ----
    axes[2].plot(
        df3[X_COL], df3[Y3_COL],
        linewidth=LINE_WIDTH_3,
        color=LINE_COLOR_3,
        linestyle=LINE_STYLE_3
    )
    axes[2].set_title(TITLE_3, fontsize=TITLE_SIZE)
    axes[2].set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    axes[2].set_ylabel(Y3_LABEL, fontsize=LABEL_SIZE)
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    style_axes(axes[2])

    # Layout adjustment
    plt.tight_layout()
    plt.savefig("Training Plots.png", dpi=DPI, bbox_inches='tight')
    plt.show()

# =======================
# RUN
# =======================
if __name__ == "__main__":
    main()