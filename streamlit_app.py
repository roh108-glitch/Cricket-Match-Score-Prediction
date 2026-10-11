import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
REQUEST_TIMEOUT = 5  # seconds

VENUES = ["Home", "Away"]
PITCHES = ["Batting", "Bowling", "Balanced"]
WEATHERS = ["Sunny", "Cloudy", "Overcast"]


def error_message(response):
    """Turn a FastAPI error response into readable text."""
    try:
        detail = response.json().get("detail", "")
    except ValueError:
        return f"Unexpected response from the API (status {response.status_code})."

    # Validation errors (422) come back as a list of problems
    if isinstance(detail, list):
        return "; ".join(item.get("msg", "Invalid input") for item in detail)
    return str(detail) or f"Request failed with status {response.status_code}."


st.set_page_config(page_title="T20 Score Predictor", page_icon="🏏")
st.title("T20 Cricket Final Score Predictor")
st.write("Enter the current match situation to estimate the team's final score.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        overs_played = st.number_input("Overs played", min_value=0, max_value=20, value=15, step=1)
        wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=10, value=7, step=1)
        run_rate = st.number_input("Current run rate", min_value=0.0, value=8.5, step=0.1, format="%.2f")
        oppo_strength = st.number_input("Opponent strength (1 to 9)", min_value=1, max_value=9, value=5, step=1)

    with col2:
        venue = st.selectbox("Venue", VENUES)
        pitch_condition = st.selectbox("Pitch condition", PITCHES)
        weather_condition = st.selectbox("Weather", WEATHERS)

    submitted = st.form_submit_button("Predict final score")

if submitted:
    payload = {
        "overs_played": int(overs_played),
        "wickets_lost": int(wickets_lost),
        "run_rate": float(run_rate),
        "venue": venue,
        "oppo_strength": int(oppo_strength),
        "pitch_condition": pitch_condition,
        "weather_condition": weather_condition,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the prediction API. Start it with: "
            "python -m uvicorn app:app --reload"
        )
    except requests.exceptions.Timeout:
        st.error("The prediction API took too long to respond. Please try again.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Something went wrong while contacting the API: {exc}")
    else:
        if response.status_code == 200:
            score = response.json()["predicted_final_score"]
            st.metric("Predicted final score", score)
            st.caption(
                "This is an estimate from a model trained on historical data. "
                "It is not a guaranteed outcome."
            )
        else:
            st.error(f"The API could not process this input: {error_message(response)}")