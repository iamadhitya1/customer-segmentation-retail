"""
Customer Segmentation for a Retail Company
-----------------------------------------
Segments retail customers into groups using K-Means clustering on their
Annual Income and Spending Score, so the business can target each group
differently.

Dataset: retail_data.csv (200 customers, "Mall Customers" dataset)
Run:     python customer_segmentation.py
Outputs: elbow_plot.png, customer_segments.png  (also shown on screen)
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_FILE = "retail_data.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42


# --- STEP 1: Load the data ---
def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Place the retail dataset next to this script."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


# --- STEP 2: Preprocessing ---
def scale_features(df):
    """Standardize the two clustering features so neither dominates the
    distance metric (income is ~15-140, spending score is ~1-100)."""
    scaler = StandardScaler()
    return scaler.fit_transform(df[FEATURES])


# --- STEP 3: Choose the number of clusters ---
def compute_wcss(X_scaled, k_range=range(1, 11)):
    """WCSS (inertia) for each k, for the elbow plot."""
    wcss = []
    for k in k_range:
        km = KMeans(
            n_clusters=k, init="k-means++", n_init=10, max_iter=300,
            random_state=RANDOM_STATE,
        )
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return list(k_range), wcss


def pick_k(X_scaled, k_range=range(2, 9)):
    """Choose k by the highest average silhouette score. This is more
    objective than eyeballing the elbow; for this dataset it agrees with
    the visible five income/spending segments (k = 5)."""
    scores = {
        k: silhouette_score(
            X_scaled,
            KMeans(n_clusters=k, init="k-means++", n_init=10,
                   random_state=RANDOM_STATE).fit_predict(X_scaled),
        )
        for k in k_range
    }
    for k, s in scores.items():
        print(f"  k={k}: silhouette={s:.3f}")
    return max(scores, key=scores.get)


def plot_elbow(k_values, wcss, chosen_k, path="elbow_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, wcss, marker="o")
    plt.axvline(chosen_k, color="red", linestyle="--", label=f"chosen k = {chosen_k}")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS (inertia)")
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


# --- STEP 4: Apply K-Means ---
def fit_kmeans(X_scaled, k):
    km = KMeans(
        n_clusters=k, init="k-means++", n_init=10, max_iter=300,
        random_state=RANDOM_STATE,
    )
    return km.fit_predict(X_scaled)


# --- STEP 5: Visualize the clusters ---
def plot_clusters(df, path="customer_segments.png"):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x=FEATURES[0], y=FEATURES[1],
        hue="Cluster", palette="viridis", s=100,
    )
    plt.title("Customer Segments")
    plt.legend(title="Cluster")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()


# --- STEP 6: Describe each cluster ---
def describe_clusters(df):
    summary = df.groupby("Cluster")[FEATURES].mean().round(1)

    def band(series, value):
        low, high = series.quantile(0.33), series.quantile(0.66)
        return "low" if value < low else "high" if value > high else "mid"

    labels = [
        f"{band(df[FEATURES[0]], row[FEATURES[0]])} income / "
        f"{band(df[FEATURES[1]], row[FEATURES[1]])} spending"
        for _, row in summary.iterrows()
    ]
    summary["Segment"] = labels
    summary["Customers"] = df["Cluster"].value_counts().sort_index().values
    return summary


def main():
    df = load_data()
    print(f"Loaded {len(df)} customers.\n")

    X_scaled = scale_features(df)

    k_values, wcss = compute_wcss(X_scaled)
    chosen_k = pick_k(X_scaled)
    print(f"Chosen k = {chosen_k}\n")
    plot_elbow(k_values, wcss, chosen_k)

    df["Cluster"] = fit_kmeans(X_scaled, chosen_k)
    plot_clusters(df)

    print("Cluster summary:")
    print(describe_clusters(df).to_string())


if __name__ == "__main__":
    main()
