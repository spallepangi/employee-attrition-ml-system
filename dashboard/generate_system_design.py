"""
Generate a professional system design / architecture diagram for the project.
Run once to create assets/system_design.png; the dashboard displays it.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "system_design.png"


def draw_diagram():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Colors
    data_c = "#4A90D9"
    process_c = "#50C878"
    model_c = "#E07C24"
    serve_c = "#9B59B6"
    arrow_c = "#2C3E50"

    def box(ax, x, y, w, h, label, color, fontsize=10):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize, color="white", weight="bold", wrap=True)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=arrow_c, lw=2))

    # Title
    ax.text(5, 11.2, "Employee Attrition Prediction System — Architecture", ha="center", fontsize=14, weight="bold")
    ax.text(5, 10.7, "Data → Preprocessing → Training → Serving", ha="center", fontsize=10, color="gray")

    # Layer 1: Data
    box(ax, 0.5, 9, 4, 1.2, "Raw Data\n(IBM HR CSV)", data_c, 9)
    box(ax, 5.2, 9, 4, 1.2, "Validation\n(Schema & missing)", data_c, 9)

    # Layer 2: Preprocessing
    box(ax, 1, 6.8, 2.8, 1.2, "Feature\nEngineering", process_c, 9)
    box(ax, 4.2, 6.8, 2.8, 1.2, "Encode &\nScale", process_c, 9)
    box(ax, 7.4, 6.8, 2.8, 1.2, "Train/Test\nSplit", process_c, 9)

    # Layer 3: Training
    box(ax, 2, 4.5, 2, 1.2, "Logistic\nRegression", model_c, 8)
    box(ax, 4.5, 4.5, 2, 1.2, "Random\nForest", model_c, 8)
    box(ax, 7, 4.5, 2, 1.2, "XGBoost\n(Optuna)", model_c, 8)

    # Layer 4: Best model
    box(ax, 3.5, 2.5, 3, 1.2, "Best Model (ROC AUC)\n+ Preprocessor", model_c, 9)

    # Layer 5: Serving
    box(ax, 1.5, 0.5, 2.5, 1.2, "FastAPI\n(Predict)", serve_c, 9)
    box(ax, 5.5, 0.5, 2.5, 1.2, "Streamlit\n(Dashboard)", serve_c, 9)

    # Arrows
    arrow(ax, 2.5, 9.6, 2.4, 8)
    arrow(ax, 7.2, 9.6, 8.2, 8)
    arrow(ax, 2.4, 7.4, 2.4, 7)
    arrow(ax, 5.5, 7.4, 5.5, 7)
    arrow(ax, 8.8, 7.4, 8.8, 7)
    arrow(ax, 2.4, 6.8, 3, 5.7)
    arrow(ax, 5.5, 6.8, 5.5, 5.7)
    arrow(ax, 8.8, 6.8, 8, 5.7)
    arrow(ax, 3, 4.5, 4, 3.7)
    arrow(ax, 5.5, 4.5, 5, 3.7)
    arrow(ax, 8, 4.5, 6, 3.7)
    arrow(ax, 5, 2.5, 2.75, 1.7)
    arrow(ax, 5, 2.5, 6.75, 1.7)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    draw_diagram()
