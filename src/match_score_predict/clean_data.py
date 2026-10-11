import pandas as pd

from .config import DATA_PATH


def clean_data(df):
    df = df.copy()  # work on a copy so the caller's DataFrame is not changed

    # remove leading/trailing white spaces from text columns
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].str.strip()

    # drop the unique id column (not a useful feature)
    if 'Match ID' in df.columns:
        df = df.drop(columns=['Match ID'])

    # rename target column
    df = df.rename(columns={'Predicted Score': 'Score'})

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    cleaned = clean_data(df)
    print(cleaned.head())
    print(cleaned.columns.tolist())


if __name__ == "__main__":
    main()