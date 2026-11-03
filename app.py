
from flask import Flask, render_template, request
import joblib
import re
import os

app = Flask(__name__)

# Get the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved machine learning model and vectorizer
model = joblib.load(os.path.join(BASE_DIR, "spam_classifier.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))


# Clean email text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text




# Website home page
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    ham_probability = None
    spam_probability = None
    email_text = ""

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()

        if email_text:
            cleaned_email = clean_text(email_text)

            # Convert email text into TF-IDF features
            email_features = tfidf.transform([cleaned_email])

            # Predict the email category
            prediction = model.predict(email_features)[0]

            # Calculate prediction probabilities
            probabilities = model.predict_proba(email_features)[0]
            probability_map = dict(zip(model.classes_, probabilities))

            ham_probability = round(
                probability_map.get("ham", 0) * 100, 2
            )

            spam_probability = round(
                probability_map.get("spam", 0) * 100, 2
            )

    return render_template(
        "index.html",
        prediction=prediction,
        ham_probability=ham_probability,
        spam_probability=spam_probability,
        email_text=email_text
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)

