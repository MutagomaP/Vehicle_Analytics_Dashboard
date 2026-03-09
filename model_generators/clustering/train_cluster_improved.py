import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

# Numerical features used to learn segments and compute a meaningful Silhouette Score.
# We focus clustering on income and price, which tend to form clearer, well-separated groups.
SEGMENT_FEATURES = ["estimated_income", "selling_price"]
CLUSTER_FEATURES = SEGMENT_FEATURES

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
df["income_level"] = df["income_level"].astype(str).str.strip().str.lower()

# Improved approach: cluster on multiple relevant numerical features so that
# K-means discovers natural groupings instead of just rediscovering a label.
preprocess = StandardScaler()


def _build_pipeline(n_clusters: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=50,
                    max_iter=300,
                    algorithm="lloyd",
                ),
            ),
        ]
    )


# Try a small range of cluster numbers and keep the configuration
# with the highest Silhouette Score.
best_score = -1.0
best_labels = None
best_pipeline: Pipeline | None = None

for k in range(2, 7):  # try between 2 and 6 clusters
    pipeline = _build_pipeline(k)
    labels = pipeline.fit_predict(df[CLUSTER_FEATURES])
    X_scaled = pipeline.named_steps["preprocess"].transform(df[CLUSTER_FEATURES])
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_score = score
        best_labels = labels
        best_pipeline = pipeline

if best_pipeline is None or best_labels is None:
    raise RuntimeError("Failed to fit any clustering configuration.")

df["cluster_id"] = best_labels
clustering_pipeline = best_pipeline

# Use the scaled numeric feature space to compute a meaningful Silhouette Score
Xp = clustering_pipeline.named_steps["preprocess"].transform(df[CLUSTER_FEATURES])
silhouette_avg = round(silhouette_score(Xp, df["cluster_id"]), 4)

# Map clusters to human-friendly classes based on dominant income_level in each cluster
cluster_to_level = (
    df.groupby("cluster_id")["income_level"]
    .agg(lambda s: s.value_counts().index[0])
    .to_dict()
)
level_to_class = {"low": "Economy", "medium": "Standard", "high": "Premium"}
cluster_id_to_class = {cid: level_to_class.get(level, "Standard") for cid, level in cluster_to_level.items()}
df["client_class"] = df["cluster_id"].map(cluster_id_to_class)

# Compute thresholds to infer income_level from numeric income at inference time
low_max = float(df.loc[df["income_level"] == "low", "estimated_income"].max())
medium_max = float(df.loc[df["income_level"] == "medium", "estimated_income"].max())
income_level_thresholds = {"low_max": low_max, "medium_max": medium_max}

# Save artifacts
joblib.dump(clustering_pipeline, "model_generators/clustering/clustering_model.pkl")
joblib.dump(cluster_id_to_class, "model_generators/clustering/cluster_id_to_class.pkl")
joblib.dump(income_level_thresholds, "model_generators/clustering/income_level_thresholds.pkl")

# Calculate Coefficient of Variation for each feature
def calculate_cv(series):
    """Calculate coefficient of variation: (std / mean) * 100"""
    return (series.std() / series.mean()) * 100

cv_income = round(calculate_cv(df["estimated_income"]), 2)
cv_price = round(calculate_cv(df["selling_price"]), 2)

# Build cluster summary
cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

# Build comparison dataframe
comparison_df = df[["client_name", "income_level", "estimated_income", "selling_price", "client_class"]]


def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "cv_income": cv_income,
        "cv_price": cv_price,
        "cv_overall": round((cv_income + cv_price) / 2, 2),
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }
