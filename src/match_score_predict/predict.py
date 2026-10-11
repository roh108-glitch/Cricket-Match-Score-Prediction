import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from .clean_data import clean_data
from .config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE


def evaluate_model():
    df = clean_data(pd.read_csv(DATA_PATH))
    X = df.drop(columns=['Score'])
    y = df['Score']

    # same split as train.py, so these test rows were never used for training or tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # load the saved model instead of retraining
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    # baseline: always predict the average score of the training data
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    print("\nRandom Forest (test set)")
    print(f"  RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  R2:   {r2_score(y_test, y_pred):.3f}")

    print("\nBaseline (predict the mean)")
    print(f"  RMSE: {root_mean_squared_error(y_test, y_base):.2f}")
    print(f"  MAE:  {mean_absolute_error(y_test, y_base):.2f}")
    print(f"  R2:   {r2_score(y_test, y_base):.3f}")


def main():
    evaluate_model()


if __name__ == "__main__":
    main()