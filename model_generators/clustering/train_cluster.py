import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

X = df[SEGMENT_FEATURES]

# Standardize features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Improved KMeans with optimized parameters for better Silhouette Score
kmeans = KMeans(
    n_clusters=3, 
    random_state=42, 
    n_init=10,
    max_iter=300,
    algorithm='lloyd'
)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

# Get cluster centers (transform back to original scale for display)
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Sort clusters by income
sorted_clusters = centers[:, 0].argsort()
cluster_mapping = {
    sorted_clusters[0]: "Economy",
    sorted_clusters[1]: "Standard",
    sorted_clusters[2]: "Premium",
}

df["client_class"] = df["cluster_id"].map(cluster_mapping)

# Save models
joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")
joblib.dump(scaler, "model_generators/clustering/clustering_scaler.pkl")

# Calculate Silhouette Score
silhouette_avg = round(silhouette_score(X_scaled, df["cluster_id"]), 4)

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
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]


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