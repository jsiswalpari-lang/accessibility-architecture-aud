from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    data = load_iris()

    model = LogisticRegression(max_iter=200)
    model.fit(data.data, data.target)

    joblib.dump(model, "model.pkl")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
