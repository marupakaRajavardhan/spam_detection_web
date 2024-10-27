import os
import re
import nltk
import numpy as np
import string
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

nltk.download('stopwords')
train_data_directory = r'C:\Users\marup\OneDrive\Documents\train_test_mails\train-mails'
test_data_directory = r'C:\Users\marup\OneDrive\Documents\train_test_mails\test-mails'

def rename_files(data_directory):
    for filename in os.listdir(data_directory):
        if filename.startswith('spmsg'):
            new_filename = f"spam_{filename}.txt"
        else:
            new_filename = f"not_spam_{filename}.txt"
        os.rename(os.path.join(data_directory, filename), os.path.join(data_directory, new_filename))

def load_and_preprocess_data(data_directory):
    emails = []
    for filename in os.listdir(data_directory):
        with open(os.path.join(data_directory, filename), 'r', encoding='utf-8') as file:
            emails.append(file.read())
    return emails

def preprocess_data(data):
    stop_words = set(stopwords.words('english'))
    preprocessed_data = []

    for email in data:
        email = email.lower()
        email = email.translate(str.maketrans('', '', string.punctuation))
        email = ' '.join([word for word in email.split() if word not in stop_words])
        preprocessed_data.append(email)
    return preprocessed_data

def extract_features(train_data, test_data):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test, vectorizer

def train_model(X_train, train_labels):
    model = MultinomialNB()
    model.fit(X_train, train_labels)
    return model

def evaluate_model(model, X_test, test_labels):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

if __name__ == "_main_":

    rename_files(train_data_directory)
    rename_files(test_data_directory)

    train_data = load_and_preprocess_data(train_data_directory)
    test_data = load_and_preprocess_data(test_data_directory)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    train_labels = ['spam' if 'spam' in filename else 'not_spam' for filename in os.listdir(train_data_directory)]
    test_labels = ['spam' if 'spam' in filename else 'not_spam' for filename in os.listdir(test_data_directory)]

    X_train, X_test, vectorizer = extract_features(train_data, test_data)
    model = train_model(X_train, train_labels)
    evaluate_model(model, X_test, test_labels)


def load_data(data_directory):
    data = []
    labels = []
    for filename in os.listdir(data_directory):
        with open(os.path.join(data_directory, filename), 'r', encoding='utf-8') as file:
            data.append(file.read())
            labels.append(1 if 'spam_spm' in filename else 0)
    return data, labels

train_data, train_labels = load_data(train_data_directory)
test_data, test_labels = load_data(test_data_directory)

def preprocess_data(data):
    stop_words = set(stopwords.words('english'))
    preprocessed_data = []
    
    for email in data:
        email = email.lower()
        email = re.sub(r'\d+', '', email)
        email = re.sub(r'[^\w\s]', '', email)
        email = ' '.join([word for word in email.split() if word not in stop_words])
        preprocessed_data.append(email)
    return preprocessed_data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

print("Data loaded and preprocessed successfully.")

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

classifiers = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

for name, clf in classifiers.items():

    clf.fit(X_train, train_labels)
    
    predictions = clf.predict(X_test)
    
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    
    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
y_train = train_labels


model = MultinomialNB()
model.fit(X_train, y_train)

with open('spam_detection.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved.")

model_filename = 'spam_detection.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        email_text = request.form['email_text']
        preprocessed_text = preprocess_text(email_text)
        prediction = model.predict([preprocessed_text])
        result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', result=result)