# api/app.py
import sys
import os

# Get the path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the project root to the Python path
sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from src.models.predictor import Predictor
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
import json

app = Flask(
    __name__, template_folder="../web_ui/templates", static_folder="../web_ui/static"
)
CORS(app)

# Initialize components
predictor = Predictor("models/saved_models/titanic_model.pkl")
cleaner = DataCleaner()
engineer = FeatureEngineer()


@app.route("/")
def index():
    """Serve the main UI"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_single():
    """Predict survival for a single passenger"""
    try:
        data = request.json
        print(f"Received data: {data}")  # Debug logging

        # Validate required fields
        required_fields = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])
        print(f"DataFrame created: {df.dtypes}")  # Debug logging

        # Process data
        df_cleaned = cleaner.clean_data(df)
        df_engineered = engineer.engineer_features(df_cleaned)
        df_encoded = engineer.encode_categorical(df_engineered)

        # Prepare features - get all possible feature columns from training
        base_features = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "FamilySize",
            "IsAlone",
        ]

        # All possible one-hot encoded columns that could exist
        all_possible_features = base_features + [
            "Embarked_C",
            "Embarked_Q",
            "Embarked_S",
            "Title_Master",
            "Title_Miss",
            "Title_Mr",
            "Title_Mrs",
            "Title_Rare",
            "AgeGroup_Child",
            "AgeGroup_Teen",
            "AgeGroup_Adult",
            "AgeGroup_Middle",
            "AgeGroup_Senior",
            "FareGroup_Low",
            "FareGroup_Medium",
            "FareGroup_High",
            "FareGroup_Very High",
        ]

        # Create a DataFrame with all possible columns, initialized to 0
        feature_df = pd.DataFrame(0, index=[0], columns=all_possible_features)

        # Fill in the values we have from df_encoded
        for col in df_encoded.columns:
            if col in all_possible_features:
                feature_df[col] = df_encoded[col].iloc[0]

        X = feature_df
        print(f"Final features shape: {X.shape}")  # Debug logging
        print(f"Final feature columns: {X.columns.tolist()}")  # Debug logging

        # Make prediction
        if predictor.model is None:
            return jsonify(
                {"error": "Model not loaded. Please train the model first."}
            ), 500

        prediction = predictor.predict(X)[0]
        probability = predictor.predict_proba(X)[0]

        return jsonify(
            {
                "prediction": int(prediction),
                "survival_probability": float(probability[1]),
                "death_probability": float(probability[0]),
                "status": "Survived" if prediction == 1 else "Did not survive",
            }
        )

    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Debug logging
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Predict survival for multiple passengers"""
    try:
        data = request.json
        passengers = data.get("passengers", [])
        filters = data.get("filters", {})
        sort_by = data.get("sort_by", "survival_probability")
        ascending = data.get("ascending", False)

        # Convert to DataFrame
        df = pd.DataFrame(passengers)

        # Process data
        df_cleaned = cleaner.clean_data(df)
        df_engineered = engineer.engineer_features(df_cleaned)
        df_encoded = engineer.encode_categorical(df_engineered)

        # Prepare features
        feature_columns = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "FamilySize",
            "IsAlone",
        ] + [
            col
            for col in df_encoded.columns
            if col.startswith(("Embarked_", "Title_", "AgeGroup_", "FareGroup_"))
        ]

        # Handle missing columns
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        X = df_encoded[feature_columns]

        # Make predictions
        predictions = predictor.predict(X)
        probabilities = predictor.predict_proba(X)

        # Apply filters and sorting
        if filters:
            df_filtered, predictions_filtered = predictor.filter_predictions(
                df_encoded, predictions, filters
            )
            probabilities_filtered = probabilities[df_filtered.index]
        else:
            df_filtered = df_encoded
            predictions_filtered = predictions
            probabilities_filtered = probabilities

        if sort_by:
            df_sorted, predictions_sorted = predictor.sort_predictions(
                df_filtered, predictions_filtered, sort_by, ascending
            )
            probabilities_sorted = (
                probabilities_filtered[df_sorted.index]
                if hasattr(df_sorted, "index")
                else probabilities_filtered
            )
        else:
            df_sorted = df_filtered
            predictions_sorted = predictions_filtered
            probabilities_sorted = probabilities_filtered

        # Prepare results
        results = []
        for i, (_, row) in enumerate(
            df_sorted.iterrows()
            if hasattr(df_sorted, "iterrows")
            else enumerate(df_sorted)
        ):
            if hasattr(row, "to_dict"):
                passenger_data = row.to_dict()
            else:
                passenger_data = dict(row) if isinstance(row, dict) else {}

            results.append(
                {
                    "passenger_data": passenger_data,
                    "prediction": int(predictions_sorted[i]),
                    "survival_probability": float(probabilities_sorted[i][1])
                    if len(probabilities_sorted) > i
                    else 0.0,
                    "status": "Survived"
                    if predictions_sorted[i] == 1
                    else "Did not survive",
                }
            )

        return jsonify(
            {
                "results": results,
                "total_count": len(results),
                "survival_rate": float(np.mean(predictions_sorted)),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/filters", methods=["GET"])
def get_available_filters():
    """Get available filter options"""
    return jsonify(
        {
            "pclass": [1, 2, 3],
            "sex": ["male", "female"],
            "embarked": ["C", "Q", "S"],
            "age_range": {"min": 0, "max": 100},
            "survival_prediction": [0, 1],
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": predictor.model is not None})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
