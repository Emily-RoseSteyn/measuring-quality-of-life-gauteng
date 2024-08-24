import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from utils.env_variables import RESULTS_DIR


def plot_actual_vs_predicted(df: pd.DataFrame, split: str):
    years = df["year"].unique()
    for year in years:
        # Actual vs predicted
        y_actual = df[df["year"] == year]["actual"]
        y_pred = df[df["year"] == year]["predicted"]
        plt.scatter(y_actual, y_pred, marker="o")

        # Trend line
        z = np.polyfit(y_actual, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_actual, p(y_actual), "r--")

        # r squared
        r2 = r2_score(y_actual, y_pred)
        r2 = f"r^2={round(r2, 2)}"
        plt.gca().text(0.05, 0.05, r2, transform=plt.gca().transAxes,
                       fontsize=8, verticalalignment="top")

        # Labelling
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted for {year}")
        plt.savefig(f"{RESULTS_DIR}/actual_vs_predicted_{split}_{year}.png")
        plt.close()
