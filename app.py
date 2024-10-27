from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

model_filename = r'C:\Users\marup\OneDrive\Documents\spam_detection\spam_detection.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

vectorizer_filename = r'C:\Users\marup\OneDrive\Documents\spam_detection\vectorizer.pkl'
with open(vectorizer_filename, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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
    
        vectorized_text = vectorizer.transform([preprocessed_text])

        prediction = model.predict(vectorized_text)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
