import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def features_targets(df):
    X = df[['parentalEdu','internet','studyHours','attendance','prevPerc']].copy()
    # scale percentages to 0-1
    X['attendance'] = X['attendance'] / 100.0
    X['prevPerc'] = X['prevPerc'] / 100.0
    y = df['presentPerc'] / 100.0
    return X, y

if __name__ == "__main__":
    df = load_data(r"C:\Users\shaik abdulrasool\Downloads\student_performance.csv")
    X, y = features_targets(df)
    print("Features shape:", X.shape)
    print(X.head())