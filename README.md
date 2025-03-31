# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY: CODTECH IT SOLUTIONS

NAME: MODIBOINA CHANDRA KEERTHI

INTERN ID: CT08WB45

DOMAIN: PYTHON PROGRAMMING

MENTOR:NEELA SANTHOSH

DESCRIPTION:

Spam Detection using Naïve Bayes Classifier

Introduction

Spam detection is a crucial application of machine learning in the field of natural language processing (NLP). In this project, we use a Naïve Bayes classifier to classify messages as either spam or ham (non-spam). The model is trained using a dataset of labeled text messages and predicts whether a given message is spam or not.
Libraries Used
The following Python libraries are used in this project:
pandas: For handling and processing tabular data.
numpy: For numerical computations.
seaborn and matplotlib.pyplot: For data visualization.
sklearn.model_selection: For splitting the dataset into training and testing sets.
sklearn.feature_extraction.text: For text vectorization using CountVectorizer and TfidfTransformer.
sklearn.naive_bayes: For implementing the Multinomial Naïve Bayes classifier.
sklearn.pipeline: For building a pipeline that automates text processing and classification.
sklearn.metrics: For evaluating the model's performance using accuracy, classification report, and confusion matrix.
Dataset Loading and Preprocessing
The dataset is loaded from a CSV file stored in the local directory. It contains labeled text messages with two primary columns:
v1: The label (spam or ham).
v2: The text message content.
The script processes the dataset as follows:
Reads the CSV file using pandas.read_csv().
Displays the first few rows and column names for verification.
Retains only the relevant columns and renames them:
v1 → label
v2 → message
Encodes the labels:
ham → 0
spam → 1
Splitting Data into Training and Testing Sets
The dataset is divided into training and testing sets using train_test_split() with an 80-20 split:
X_train, y_train: Training set.
X_test, y_test: Testing set.
Building the Text Classification Pipeline
A Pipeline is created to automate the text preprocessing and classification process. It consists of three stages:
Vectorization (CountVectorizer): Converts text into a bag-of-words representation.
TF-IDF Transformation (TfidfTransformer): Converts word frequency counts into TF-IDF values to account for the importance of words.
Classification (MultinomialNB): Uses a Naïve Bayes classifier trained on the transformed text data.
Model Training
The pipeline is trained using pipeline.fit(X_train, y_train), where it learns to distinguish between spam and ham messages based on textual patterns.
Making Predictions
The trained model makes predictions on the test set using pipeline.predict(X_test), generating a list of predicted labels (spam or ham).
Evaluating the Model
The model's performance is assessed using:
Accuracy Score (accuracy_score): Measures the proportion of correctly classified messages.
Classification Report (classification_report): Provides precision, recall, and F1-score for spam and ham classifications.
Confusion Matrix (confusion_matrix): Displays the number of correct and incorrect predictions for each class.
The confusion matrix is visualized using seaborn.heatmap() with annotations to highlight correct and incorrect classifications.
User Interaction for Message Prediction
The script includes a loop that allows users to input their own text messages for classification. The user can enter a message, and the model will classify it as spam or ham. The loop continues until the user types 'exit'.
Summary
This project demonstrates how to build a spam detection system using a Naïve Bayes classifier. The model preprocesses text messages, learns patterns from labeled data, and classifies new messages based on learned features. The combination of CountVectorizer, TF-IDF transformation, and Naïve Bayes provides an efficient method for spam filtering.
The approach can be extended further by:
Using more advanced NLP techniques (e.g., word embeddings, deep learning models).
Improving feature extraction by incorporating additional text processing steps.
Fine-tuning the model with different classifiers or hyperparameter tuning.

OUTPUT:

![Image](https://github.com/user-attachments/assets/e2a99c45-14d2-427d-8fc6-f6264a487bb7)

![Image](https://github.com/user-attachments/assets/4d749a9d-a84d-4df7-92d5-f2cc26a06fea)
