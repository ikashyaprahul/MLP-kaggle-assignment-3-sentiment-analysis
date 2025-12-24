import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, save_model
from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    SAMPLE_SUBMISSION_PATH,
    TEXT_COL,
    TARGET_COL,
    MODEL_PATH,
)


def main():
    # Load data
    train_df, _, _ = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH)

    # Preprocess data
    train_df = preprocess_data(train_df, TEXT_COL, TARGET_COL)

    # Train model
    model = train_model(train_df, train_df[TARGET_COL], TEXT_COL)

    # Save model
    save_model(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
