import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import load_model
from src.config import (
    TEST_DATA_PATH,
    SAMPLE_SUBMISSION_PATH,
    TEXT_COL,
    MODEL_PATH,
    SUBMISSION_PATH,
    TRAIN_DATA_PATH,
)


def main():
    # Load data
    _, test_df, submission_df = load_data(
        TRAIN_DATA_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH
    )

    # Preprocess data
    test_df = preprocess_data(test_df, TEXT_COL)

    # Load model
    model = load_model(MODEL_PATH)

    # Make predictions
    predictions = model.predict(test_df[TEXT_COL])

    # Create submission file
    submission_df.iloc[:, 1] = predictions
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
