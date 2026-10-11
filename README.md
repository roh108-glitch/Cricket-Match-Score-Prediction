# T20 Cricket Final Score Predictor

A machine learning project that estimates a T20 team's final score from the match situation at a given point. It includes a training pipeline, a FastAPI prediction service, a command-line tool, and a Streamlit frontend that calls the API.

## Features

- Random Forest regression pipeline with one-hot encoding for categorical features
- Hyperparameter tuning with GridSearchCV and 5-fold cross-validation
- Held-out test set and a mean-prediction baseline for comparison
- FastAPI endpoint (`POST /predict`) with input validation and range checks
- Command-line tool with input validation
- Streamlit frontend that calls the API

## Results

Evaluated on a 20% held-out test set (300 rows):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest (tuned) | 8.74 | 6.64 | 0.985 |
| Baseline (predict the mean) | 71.40 | 59.04 | -0.001 |

Best parameters from GridSearchCV: `n_estimators=200`, `max_depth=None`, `min_samples_split=2`. Cross-validation RMSE on the training data was 9.67.

## Dataset

- File: `data/t20_cricket_match_score_prediction.csv`
- 1,500 rows and 9 columns
- Features: overs played, wickets lost, run rate, venue (Home/Away), opponent strength (1 to 9), pitch condition (Batting/Bowling/Balanced), weather (Sunny/Cloudy/Overcast)
- Target: `Score`
- Source: [add where the dataset came from]


## Project Structure

```
cricket_match_score_prediction/
├── app.py                      # FastAPI prediction API
├── streamlit_app.py            # Streamlit frontend (calls the API)
├── take_input_and_predict.py   # Command-line prediction tool
├── pyproject.toml
├── data/
│   └── t20_cricket_match_score_prediction.csv
├── model/
│   └── random_forest_model.pkl # Trained model
└── src/
    └── match_score_predict/
        ├── config.py           # Paths and settings
        ├── clean_data.py       # Cleaning steps
        ├── train.py            # Training and tuning
        └── predict.py          # Evaluation against the baseline
```

## Setup

Requires Python 3.10 or newer.

From the project root:

```bash
python -m pip install -e .
python -m pip install streamlit requests
```

## Usage

### 1. Train the model (optional)

A trained model is already included in `model/`. To retrain:

```bash
python -m match_score_predict.train
```

### 2. Evaluate the model

```bash
python -m match_score_predict.predict
```

### 3. Run the API

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the endpoint. Example request body:

```json
{
  "overs_played": 15,
  "wickets_lost": 7,
  "run_rate": 8.5,
  "venue": "home",
  "oppo_strength": 5,
  "pitch_condition": "balanced",
  "weather_condition": "sunny"
}
```

### 4. Run the Streamlit frontend

Keep the API running in one terminal, then in a second terminal:

```bash
python -m streamlit run streamlit_app.py
```

Streamlit opens at `http://localhost:8501`.

### 5. Command-line tool

```bash
python take_input_and_predict.py
```

## Limitations and Next Steps

- Some data may be synthetic, so the metrics describe performance on this dataset only.
- The model doesn't account for player-level information, team strength beyond a single number, or ball-by-ball data.
- Possible improvements: compare other models (for example, gradient boosting), add more real match data, and deploy the API and frontend online.