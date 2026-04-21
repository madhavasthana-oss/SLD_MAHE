import pandas as pd
import matplotlib.pyplot as plt

# =======================
# GLOBAL CONFIG (EDIT THESE)
# =======================

# File paths
CSV_FILE_1 = "train_accuracy.csv"
CSV_FILE_2 = "train_loss.csv"

# Column names (edit if needed)
X_COL = "Step"
Y1_COL = "train_accuracy"
Y2_COL = "train_loss"

# Titles and labels
TITLE_1 = "Training Accuracy"
TITLE_2 = "Training Loss"
X_LABEL = "Step"
Y1_LABEL = "Accuracy"
Y2_LABEL = "Loss"

# Font settings
FONT_FAMILY = "STIXGeneral"
TITLE_SIZE = 18
LABEL_SIZE = 14
TICK_SIZE = 12

# Line styling
LINE_WIDTH_1 = 1.2
LINE_WIDTH_2 = 1.2

LINE_COLOR_1 = "black"
LINE_COLOR_2 = "blue"

LINE_STYLE_1 = "-"
LINE_STYLE_2 = "-"

# Figure and subplot size
FIG_SIZE = (12, 5)   # whole figure
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

    # Create subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI)

    # ---- Plot 1 (0,0) ----
    axes[0].plot(
        df1[X_COL],
        df1[Y1_COL],
        linewidth=LINE_WIDTH_1,
        color=LINE_COLOR_1,
        linestyle=LINE_STYLE_1
    )

    axes[0].set_title(TITLE_1, fontsize=TITLE_SIZE)
    axes[0].set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    axes[0].set_ylabel(Y1_LABEL, fontsize=LABEL_SIZE)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    style_axes(axes[0])

    # ---- Plot 2 (0,1) ----
    axes[1].plot(
        df2[X_COL],
        df2[Y2_COL],
        linewidth=LINE_WIDTH_2,
        color=LINE_COLOR_2,
        linestyle=LINE_STYLE_2
    )

    axes[1].set_title(TITLE_2, fontsize=TITLE_SIZE)
    axes[1].set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    axes[1].set_ylabel(Y2_LABEL, fontsize=LABEL_SIZE)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    style_axes(axes[1])

    # Layout adjustment
    plt.tight_layout()

    # Show plot
    plt.show()

# =======================
# RUN
# =======================

if __name__ == "__main__":
    main()