import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .clean_data import clean_data
from .config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE


def train_model(df):
    df = clean_data(df)

    # features and target
    X = df.drop(columns=['Score'])
    y = df['Score']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=RANDOM_STATE)),
    ])

    # hold out the test set first; it is not used for tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
    }

    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    print(f"\nBest cross-validation RMSE: {np.sqrt(-grid_search.best_score_):.2f}")

    return grid_search.best_estimator_, X_test, y_test


def main():
    df = pd.read_csv(DATA_PATH)
    model, X_test, y_test = train_model(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()