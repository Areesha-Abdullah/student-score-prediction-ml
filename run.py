from src.train import train_linear_model, train_polynomial_model
from src.evaluate import evaluate_model, plot_predictions


DATA_PATH = "data/student_data.csv"

# Linear Regression
y_pred, y_test = train_linear_model(DATA_PATH)
evaluate_model(y_test, y_pred, model_name="Linear Regression")
plot_predictions(y_test, y_pred, model_name="Linear Regression")

# Polynomial Regression
y_test_poly, y_pred_poly = train_polynomial_model(DATA_PATH, degree=2)
evaluate_model(y_test_poly, y_pred_poly, model_name="Polynomial Regression")        
plot_predictions(y_test_poly, y_pred_poly, model_name="Polynomial Regression")  
