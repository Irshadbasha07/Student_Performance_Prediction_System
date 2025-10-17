import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_preprocessing import load_data, features_targets

def train_and_save(model_path="saved_models/student_model.pkl"):
    df = load_data(r"C:\Users\shaik abdulrasool\Downloads\student_performance.csv")
    X, y = features_targets(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print("Saved model to", model_path)
    return model

if __name__ == "__main__":
    train_and_save()