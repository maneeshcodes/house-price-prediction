from pathlib import Path
import joblib



def save_model(model, filename="my_california_housing_model.pkl"):
    Path("../models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path("../models") / filename)