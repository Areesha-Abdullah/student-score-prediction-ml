import pandas as pd
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()  # Remove rows with missing values

    #define target columns
    target_column = "Exam_Score"

    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target variable

    #encode categoricsal variables

    X = pd.get_dummies(X, drop_first=True)

    return X,y


