import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,  mean_squared_error, r2_score

def evaluate_model(y_true , y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    return mae, rmse, r2


def plot_predictions(y_true, y_pred, model_name="Model"):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Exam Scores")
    plt.ylabel("Predicted Exam Scores")
    plt.title(f"{model_name} Predictions vs Actual")
    plt.show()
