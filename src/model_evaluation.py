import joblib
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_data, features_targets

def evaluate(model_path="saved_models/student_model.pkl"):
    df = load_data(r"C:\Users\shaik abdulrasool\Downloads\student_performance.csv")
    X, y = features_targets(df)
    model = joblib.load(model_path)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")
    return mse, r2

if __name__ == "__main__":
    evaluate()