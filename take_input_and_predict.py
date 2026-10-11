import joblib
import pandas as pd

from match_score_predict.config import MODEL_PATH

VENUES = ["Home", "Away"]
PITCHES = ["Batting", "Bowling", "Balanced"]
WEATHERS = ["Sunny", "Cloudy", "Overcast"]


def ask_int(prompt, low, high):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Please enter a whole number.")
            continue
        if low <= value <= high:
            return value
        print(f"Please enter a number between {low} and {high}.")


def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a number, for example 8.5.")


def ask_choice(prompt, choices):
    while True:
        value = input(prompt).strip().capitalize()
        if value in choices:
            return value
        print(f"Please enter one of: {', '.join(choices)}")


def take_input():
    overs_played = ask_int("Enter overs played (0-20): ", 0, 20)
    wickets_lost = ask_int("Enter wickets lost (0-10): ", 0, 10)
    run_rate = ask_float("Enter current run rate: ")
    venue = ask_choice("Enter venue (Home/Away): ", VENUES)
    oppo_strength = ask_int("Enter opponent strength (1-9): ", 1, 9)
    pitch_condition = ask_choice("Enter pitch condition (Batting/Bowling/Balanced): ", PITCHES)
    weather_condition = ask_choice("Enter weather condition (Sunny/Cloudy/Overcast): ", WEATHERS)

    return pd.DataFrame({
        'Overs Played': [overs_played],
        'Wickets Lost': [wickets_lost],
        'Run Rate': [run_rate],
        'Home/Away': [venue],
        'Opponent Strength': [oppo_strength],
        'Pitch Condition': [pitch_condition],
        'Weather': [weather_condition],
    })


def prediction():
    input_df = take_input()
    model = joblib.load(MODEL_PATH)
    predicted_score = model.predict(input_df)
    print(f"\nPredicted Final Score: {int(predicted_score[0])}")


def main():
    prediction()


if __name__ == "__main__":
    main()