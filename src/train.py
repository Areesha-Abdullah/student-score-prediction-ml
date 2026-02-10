from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.preprocessing import load_and_preprocess_data

def train_linear_model(data_path):
    X,y = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return y_pred, y_test


def train_polynomial_model(data_path, degree=2):
    X,y = load_and_preprocess_data(data_path)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size = 0.2, random_state = 42

    )

    model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred, y_test



