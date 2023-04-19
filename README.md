# Uso de algoritmos no supervisados para identificación de fake news en tweets noticiosos previos al primer confinamiento obligatorio en Bogotá.

Here are described the steps to execute the two python scripts for this work, `thesis_training.py` and `thesis_trained.py`. The first script trains multiple classifiers on a dataset and saves the model with best performance. The second script loads the saved model and perform predictions on new data.

## Table of contents
1. [Dependencies](#dependencies)
2. [thesis_training: Training Multiple Dlassifiers](#thesis_training)
    - [Usage](#thesis_training_usage)
    - [Code Explanation](#thesis_training_code_explanation)
3. [thesis_trained: Predicting Labels Using Trained Model](#thesis_trained)
    - [Usage](#thesis_trained_usage)
    - [Code Explanation](#thesis_trained_code_explanation)

## 1. Dependencies <a name="dependencies"></a>

Before running the scripts, make sure to install the necessary dependencies, depending on the script you're running you need to install a set of libraries, if you are going to use the `thesis_training.py` run the command for `requirements.txt`, otherwise run the command for `requirements_min.py`.
```bash
pip install -r requirements.txt
```
Also run this commands for the required spacy model and to upgrade your python packages.
```bash
pip install -U pip setuptools wheel
python -m spacy download es_core_news_sm
```

## 2. thesis_training: Training Multiple Classifiers <a name="thesis_training"></a>

This script is responsible for the training of multiple classifiers on a dataset and saving the best performing model.

### Usage <a name="thesis_training_usage"></a>

To execute this script simply run:
```bash
python thesis_training.py
```

### Code Explanation <a name="thesis_training_code_explanation"></a>

This script performs the following steps:
1. Loads the dataset and process it.
2. Splits the dataset into training and testing sets.
3. Train the multiple classifiers: Multinomial Naive Bayes, Logistic Regression, SVM, and Random Forest
4. Save the best-performing model (Logistic Regression in this case) using `joblib`.

## 3. thesis_trained: Predicting Labels Using Trained Model <a name="thesis_trained"></a>

This script loads the previously saved model and performs predictions on new data.

### Usage <a name="thesis_trained_usage"></a>

To execute this script simply run:
```bash
python thesis_trained.py
```

### Code Explanation <a name="thesis_trained_code_explanation"></a>

This script performs the following steps:
1. Loads the trained model using `joblib`.
2. Read and concatenate multiple CSV files from the specified folder.
3. Process and clean the text data.
4. Perform predictions using the loaded model.
5. Save the predicted labels in a new CSV file.