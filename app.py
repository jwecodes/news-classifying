from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
with open('svc_linear.pk', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer (if used for text preprocessing)
with open('TfIdf_Vectorizer.pk', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to classify content using the model
def classify_content(content):
    # Preprocess the text (if necessary) and classify it
    content_vectorized = vectorizer.transform([content])  # Transform text using the vectorizer
    prediction = model.predict(content_vectorized)  # Predict the category using the model
    return prediction[0]  # Return the first (and only) prediction

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/classify", methods=["GET", "POST"])
def classify():
    if request.method == "POST":
        # Handle text classification
        text_input = request.form.get("text_input")
        if text_input:
            # Classify the pasted text
            result = classify_content(text_input)
            return render_template("results.html", results=f"Classified as: {result}")
        
        # Handle file upload
        file = request.files.get("news_file")
        if file:
            if file.filename.endswith('.csv'):
                # Read CSV file
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                # Read Excel file
                df = pd.read_excel(file)
            else:
                return "Unsupported file format", 400
            
            # Assuming the file contains a 'text' column with news content
            if 'text' in df.columns:
                # Apply classification to each row in the 'text' column
                df['classification'] = df['text'].apply(classify_content)
                # Example: Show classification for the first row
                result = df.iloc[0]['classification']
                return render_template("results.html", results=f"First article classified as: {result}")
            else:
                return "No 'text' column found in the file", 400

    return render_template("classify.html")

@app.route("/results")
def results():
    return render_template("results.html", results="Your results will be displayed here.")

if __name__ == "__main__":
    app.run(debug=True)
