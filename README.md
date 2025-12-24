# Kaggle Assignment 3 – Sentiment Analysis

This repository contains a refactored solution for **Kaggle Assignment 3** of the **Machine Learning Practice (MLP)** course, IIT Madras.

The objective of this assignment is to perform **multi-class sentiment analysis** on movie review phrases using **classical machine learning models** and **TF-IDF-based text features**.

---

## Project Structure

The project has been refactored from a single Jupyter Notebook into a modular structure:

- `src/`: Contains Python modules for data preprocessing (`data_preprocessing.py`), model training (`model_training.py`), and configuration (`config.py`).
- `scripts/`: Contains executable scripts for training (`train.py`) and prediction (`predict.py`).
- `notebook/`: Contains the refactored Jupyter Notebook, which now serves as a high-level interface and for visualizations.
- `data/`: (Not included in repo) This is where the Kaggle dataset files should be placed.
- `requirements.txt`: Lists the Python dependencies for the project.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/mlp-term-3-2025-kaggle-assignment-3/data) and place the `train.csv`, `test.csv`, and `sample_submission.csv` files into a `/kaggle/input/mlp-term-3-2025-kaggle-assignment-3` directory.

---

## Usage

### Training the Model

To train the sentiment analysis model, run the `train.py` script:

```bash
python scripts/train.py
```

This will preprocess the training data, train the final model (Character n-gram TF-IDF + Logistic Regression), and save the trained model to `model.joblib`.

### Generating Predictions

To generate predictions on the test set, run the `predict.py` script:

```bash
python scripts/predict.py
```

This will load the trained model, make predictions on the test data, and create a `submission.csv` file in the required Kaggle format.

---

## Author
**Kashyap Rahul**  
IIT Madras – BS Degree Programme  
Machine Learning Practice
