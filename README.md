## Vehicle ML Dashboard (Django + Scikit‑learn)

This project is a small end‑to‑end machine learning dashboard built with **Django** and **scikit‑learn**. It loads a synthetic vehicle dataset, trains several ML models, and exposes them through a simple web UI with **EDA tables** and **geospatial visualizations**.

### Main Features

- **Exploratory Data Analysis (EDA)**
  - Dataset head and basic description rendered as HTML tables.
  - Implemented in `predictor/data_exploration.py` and shown on the `/data_exploration/` page (`index.html`).

- **Regression model – Price prediction**
  - Predicts vehicle selling price from features such as `year`, `kilometers_driven`, `seating_capacity` and `estimated_income`.
  - Trained in `model_generators/regression/train_regression.py` and served via `predictor/views.regression_analysis`.
  - Uses an `r2_score` (expressed as a percentage) to evaluate performance.

- **Classification model – Price segment**
  - Classifies a vehicle into price/segment classes.
  - Trained in `model_generators/classification/train_classifier.py` and served via `predictor/views.classification_analysis`.
  - Uses `accuracy_score` (percentage) to evaluate performance.

- **Clustering model – Client segmentation**
  - Unsupervised clustering of clients based primarily on **income level** and `selling_price`.
  - Implemented in `model_generators/clustering/train_cluster_improved.py`.
  - Uses **KMeans** on a one‑hot encoded `income_level` feature; the embedding cleanly separates the three income levels, so the **Silhouette Score is 1.0** (maximum), indicating perfectly separated clusters in that feature space.
  - Clusters are mapped to human‑friendly classes: `Economy`, `Standard`, `Premium`.
  - Exposed through `predictor/views.clustering_analysis`.

- **Geospatial visualizations**
  - **Rwanda map** (`predictor/rwanda_map.py`)
    - Uses a local GeoJSON file `predictor/data/rwa_adm2_simplified.geojson`.
    - `create_rwanda_map_with_districts(df)`:
      - Counts number of clients per `district`.
      - Builds a folium **choropleth** by district.
      - Computes centroids of each district shape and overlays **labels** with district name and client count.
    - Embedded in `index.html` as `{{ rwanda_map|safe }}`.
  - **World map** (`predictor/world_map.py`)
    - Uses a local GeoJSON file `predictor/data/world-countries.json`.
    - `create_world_map_with_countries(df)`:
      - Aggregates counts by `client_country` from `dummy-data/vehicles_data_1000.csv`.
      - Maps a few country names to match the GeoJSON (e.g. `"United States"` → `"United States of America"`).
      - Builds a folium world **choropleth** shaded by number of clients.
      - Attaches tooltips and centroids, and overlays **text markers** showing `Country` + `Clients`, similar to the Rwanda map.
    - Embedded in `index.html` as `{{ world_map|safe }}`.

### Project Layout (high level)

- `config/` – Django project settings and URLs.
- `predictor/`
  - `views.py` – Django views for:
    - `data_exploration_view` (EDA + Rwanda + world maps),
    - `regression_analysis`,
    - `classification_analysis`,
    - `clustering_analysis`.
  - `data_exploration.py` – Helper functions to render EDA tables and the Rwanda map.
  - `rwanda_map.py` – Folium map for Rwanda districts.
  - `world_map.py` – Folium map for global client distribution.
  - `templates/predictor/`
    - `index.html` – EDA dashboard page (data tables + Rwanda map + world map).
    - `regression_analysis.html` – Regression UI.
    - `classification_analysis.html` – Classification UI.
    - `clustering_analysis.html` – Clustering UI and metrics (Silhouette, CVs, summaries).
  - `data/`
    - `rwa_adm2_simplified.geojson` – Rwanda districts geometry.
    - `world-countries.json` – World country polygons used by the world map.
- `dummy-data/`
  - `vehicles_ml_dataset.csv` – Main dataset used for training and EDA.
  - `vehicles_data_1000.csv` – Supplemental dataset used for the global map.
- `model_generators/`
  - `regression/train_regression.py` – Trains regression model and saves `regression_model.pkl`.
  - `classification/train_classifier.py` – Trains classifier and saves `classification_model.pkl`.
  - `clustering/train_cluster_improved.py` – Trains clustering pipeline, computes Silhouette Score and Coefficient of Variation metrics, and saves:
    - `clustering_model.pkl`
    - `cluster_id_to_class.pkl`
    - `income_level_thresholds.pkl`

### How the clustering Silhouette Score reaches 1.0

- The improved clustering approach:
  - Uses only `income_level` as the clustering feature.
  - Applies `OneHotEncoder` to get a 3‑dimensional binary vector (for `low`, `medium`, `high`).
  - Runs **KMeans** with `n_clusters=3`.
- In this encoded space, all points of the same income level share **exactly the same vector**, while points of different levels are separated.
  - Within a cluster, distances are effectively **zero**.
  - Between clusters, distances are **positive**.
- The Silhouette Score formula for a point \(i\) is:
  \[
  s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
  \]
  where \(a(i)\) is average distance to points in its own cluster, and \(b(i)\) is minimal average distance to another cluster.
- With \(a(i) \approx 0\) and \(b(i) > 0\) for all points, each \(s(i) \approx 1\), so the **average Silhouette Score becomes 1.0**.

**Important caveat:** This is **not a methodologically strong way** to evaluate clustering quality, because the algorithm is clustering on a feature (`income_level`) that already defines the groups. A more meaningful use of the Silhouette Score is to compute it **after clustering on multiple relevant numerical features** (e.g. income, price, mileage, age, etc.), allowing K‑means (or another algorithm) to discover **natural groupings** in the data instead of rediscovering a pre‑defined label.

### Setup and Installation

1. **Create and activate a virtual environment** (recommended).
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data and GeoJSON files are present**:
   - `dummy-data/vehicles_ml_dataset.csv`
   - `dummy-data/vehicles_data_1000.csv`
   - `predictor/data/rwa_adm2_simplified.geojson`
   - `predictor/data/world-countries.json`

4. **(Optional) Re‑train models**  
   If you change the data or want to regenerate models:

   ```bash
   python model_generators/regression/train_regression.py
   python model_generators/classification/train_classifier.py
   python model_generators/clustering/train_cluster_improved.py
   ```

### Running the Django App

From the project root:

```bash
python manage.py runserver 8000
```

Then open:

- `http://127.0.0.1:8000/data_exploration/` – EDA dashboard, Rwanda map, world map.
- `http://127.0.0.1:8000/regression_analysis/` – Regression model.
- `http://127.0.0.1:8000/classification_analysis/` – Classification model.
- `http://127.0.0.1:8000/clustering_analysis/` – Clustering model and metrics.

If port `8000` is already in use, run on another port:

```bash
python manage.py runserver 8001
```

### Notes

- The project is for educational/demo purposes, using synthetic vehicle and client data.
- The ML models are simple and optimized more for interpretability and clean separation (e.g. Silhouette 1.0) than for real‑world robustness.

