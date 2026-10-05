from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D

# Default to a repo-relative location (no absolute paths).
BASE_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "hpo"
VAL_CSV = "4.csv"
TEST_CSV = "4_test.csv"

VAL_METRIC = "val_f1"
TEST_METRIC = "test_f1"


def parse_params(folder_name):

    match = re.match(r"a([\d\.]+)_b([\d\.]+)_g([\d\.]+)", folder_name)

    if match is None:
        return None

    return (
        float(match.group(1)),
        float(match.group(2)),
        float(match.group(3))
    )


def collect_results():

    rows = []

    for run in BASE_DIR.rglob("a*_b*_g*"):

        params = parse_params(run.name)

        if params is None:
            continue

        alpha, beta, gamma = params

        val_csv = run / "model_weights" / VAL_CSV
        test_csv = run / "model_weights" / TEST_CSV

        if not val_csv.exists():
            continue

        val_df = pd.read_csv(val_csv)

        # find best epoch based on validation score
        best_idx = val_df[VAL_METRIC].idxmax()
        best_row = val_df.loc[best_idx]

        best_epoch = int(best_row["Epoch"])
        best_val = float(best_row[VAL_METRIC])

        test_score = np.nan

        if test_csv.exists():

            test_score = np.nan

            if test_csv.exists():

                test_df = pd.read_csv(test_csv)

                if not test_df.empty and "test_f1" in test_df.columns:

                    test_score = float(test_df["test_f1"].iloc[-1])

                else:

                    print(f"[WARN] missing test results in {test_csv}")

        rows.append({

            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,

            "best_epoch": best_epoch,

            "val_f1": best_val,
            "test_f1": test_score,

            "path": str(run)

        })

    return pd.DataFrame(rows)


def plot_top_configs(df):

    df_sorted = df.sort_values("val_f1", ascending=False)

    top = df_sorted.head(10)

    labels = [
        f"a={r.alpha},b={r.beta},g={r.gamma}"
        for _, r in top.iterrows()
    ]

    plt.figure(figsize=(10,4))

    plt.bar(range(len(top)), top.val_f1)

    plt.xticks(range(len(top)), labels, rotation=30)

    plt.ylabel("Validation Macro-F1")

    plt.title("Top Hyperparameter Configurations")

    plt.tight_layout()

    plt.savefig("hpo_top10.png", dpi=300)

    plt.close()


def plot_landscape(df, x, y):

    plt.figure(figsize=(6,5))

    sc = plt.scatter(
        df[x],
        df[y],
        c=df.val_f1,
        s=120
    )

    plt.xlabel(x)
    plt.ylabel(y)

    plt.title(f"HPO Landscape: {x} vs {y}")

    plt.colorbar(sc, label="Validation F1")

    plt.tight_layout()

    plt.savefig(f"hpo_{x}_{y}.png", dpi=300)

    plt.close()


def plot_heatmap(df):

    best_alpha = df.sort_values("val_f1", ascending=False).iloc[0]["alpha"]

    subset = df[df["alpha"] == best_alpha]

    pivot = subset.pivot_table(
        index="beta",
        columns="gamma",
        values="val_f1",
        aggfunc="mean"
    )

    plt.figure(figsize=(6,5))

    im = plt.imshow(pivot, origin="lower")

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("gamma")
    plt.ylabel("beta")

    plt.title(f"HPO Heatmap (alpha={best_alpha})")

    plt.colorbar(im, label="Validation F1")

    plt.tight_layout()

    plt.savefig("hpo_heatmap.png", dpi=300)
    plt.close()



def plot_3d_surface(df):

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')

    x = df["alpha"]
    y = df["beta"]
    z = df["gamma"]
    c = df["val_f1"]

    sc = ax.scatter(
        x,
        y,
        z,
        c=c,
        s=120
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("gamma")

    ax.set_title("Hyperparameter Search Landscape")

    fig.colorbar(sc, label="Validation F1")

    plt.tight_layout()
    plt.savefig("hpo_3d_landscape.png", dpi=300)
    plt.close()

def plot_val_vs_test(df):

    plt.figure(figsize=(6,5))

    plt.scatter(
        df["val_f1"],
        df["test_f1"],
        s=120
    )

    plt.xlabel("Validation F1")
    plt.ylabel("Test F1")

    plt.title("Validation vs Test Performance")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("hpo_val_vs_test.png", dpi=300)
    plt.close()
    

def main():

    df = collect_results()

    df = df.sort_values("val_f1", ascending=False)

    df.to_csv("hpo_summary.csv", index=False)

    print("\nBest configuration:\n")
    print(df.iloc[0])

    plot_top_configs(df)

    plot_landscape(df, "alpha", "beta")
    plot_landscape(df, "alpha", "gamma")
    plot_landscape(df, "beta", "gamma")
    plot_3d_surface(df)
    plot_heatmap(df)
    plot_val_vs_test(df)


if __name__ == "__main__":
    main()