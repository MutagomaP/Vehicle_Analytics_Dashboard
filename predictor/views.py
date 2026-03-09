from django.shortcuts import render
from model_generators.clustering.train_cluster_improved import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model
from predictor.data_exploration import dataset_exploration, data_exploration, rwanda_map_exploration
from predictor.world_map import create_world_map_with_countries
import pandas as pd
import joblib

regression_model = joblib.load("model_generators/regression/regression_model.pkl")
classification_model = joblib.load("model_generators/classification/classification_model.pkl")
clustering_model = joblib.load("model_generators/clustering/clustering_model.pkl")
cluster_id_to_class = joblib.load("model_generators/clustering/cluster_id_to_class.pkl")
income_level_thresholds = joblib.load("model_generators/clustering/income_level_thresholds.pkl")


def _income_to_level(income: float) -> str:
    low_max = income_level_thresholds.get("low_max")
    medium_max = income_level_thresholds.get("medium_max")
    if low_max is None or medium_max is None:
        return "medium"
    if income <= low_max:
        return "low"
    if income <= medium_max:
        return "medium"
    return "high"

def data_exploration_view(request):
    # Main dataset used for tables
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    # Dataset with wider world coverage used for world map
    df_world = pd.read_csv("dummy-data/vehicles_data_1000.csv")
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": rwanda_map_exploration(df),
        "world_map": create_world_map_with_countries(df_world),
    }
    
    return render(request, "predictor/index.html", context)

def regression_analysis(request):
    context = {
        "evaluations": evaluate_regression_model()
        }
    
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["price"] = prediction
    
    return render(request, "predictor/regression_analysis.html", context)




def classification_analysis(request):
    context = {
        "evaluations": evaluate_classification_model()
    }
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = prediction
        return render(request, "predictor/classification_analysis.html", context)
    
    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):
    context = {
        "evaluations": evaluate_clustering_model()
    }

    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])

            # Step 1: Predict price
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]
            # Step 2: Predict cluster
            income_level = _income_to_level(income)
            cluster_id = clustering_model.predict(pd.DataFrame({"income_level": [income_level]}))[0]

            context.update({
                "prediction": cluster_id_to_class.get(cluster_id, "Standard"),
                "price": predicted_price
            })
        except Exception as e:
            context["error"] = str(e)
    return render(request, "predictor/clustering_analysis.html", context)