from enum import Enum

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from match_score_predict.config import MODEL_PATH


class VenueOptions(str, Enum):
    home = "Home"
    away = "Away"


class PitchOptions(str, Enum):
    batting = "Batting"
    bowling = "Bowling"
    balanced = "Balanced"


class WeatherOptions(str, Enum):
    sunny = "Sunny"
    cloudy = "Cloudy"
    overcast = "Overcast"


class CricketInput(BaseModel):
    overs_played: int = Field(ge=0, le=20)
    wickets_lost: int = Field(ge=0, le=10)
    run_rate: float = Field(ge=0)
    venue: VenueOptions
    oppo_strength: int = Field(ge=1, le=9)
    pitch_condition: PitchOptions
    weather_condition: WeatherOptions

    # "home" -> "Home", "BATTING" -> "Batting", etc.
    @field_validator('venue', 'pitch_condition', 'weather_condition', mode='before')
    @classmethod
    def normalize_text(cls, v):
        if isinstance(v, str):
            return v.strip().capitalize()
        return v


app = FastAPI()
model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict_score(data: CricketInput):
    input_df = pd.DataFrame({
        'Overs Played': [data.overs_played],
        'Wickets Lost': [data.wickets_lost],
        'Run Rate': [data.run_rate],
        'Home/Away': [data.venue.value],
        'Opponent Strength': [data.oppo_strength],
        'Pitch Condition': [data.pitch_condition.value],
        'Weather': [data.weather_condition.value],
    })

    prediction = model.predict(input_df)
    return {"predicted_final_score": int(prediction[0])}