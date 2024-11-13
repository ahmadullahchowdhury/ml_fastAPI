from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class IrisModel:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.feature_names = [
            "sepal_length", "sepal_width", 
            "petal_length", "petal_width"
        ]
        
    def train(self):
        # Load iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Split the data
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.class_names = iris.target_names
        
    def predict(self, features: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
            
        prediction = self.model.predict(features)[0]
        probability = max(self.model.predict_proba(features)[0])
        
        return {
            "predicted_class": int(prediction),
            "probability": float(probability),
            "class_name": self.class_names[prediction]
        }
        
    def get_model_info(self):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
            
        return {
            "model_type": "RandomForestClassifier",
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names)
        }
