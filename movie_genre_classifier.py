# CODSOFT Internship
# Task: Movie Genre Classification
# Name: Komal Chavare

import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load training dataset
df = pd.read_csv(
    "train_data.txt",
    sep=":::",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"],
    engine="python"
)

print(df.head())
print(df.shape)

# Text Cleaning

tokenizer = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    words = tokenizer.tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Apply cleaning

df["CLEAN_DESCRIPTION"] = df["DESCRIPTION"].apply(clean_text)

# Show the result
print("\nOriginal Description:")
print(df["DESCRIPTION"].iloc[0])

print("\nCleaned Description:")
print(df["CLEAN_DESCRIPTION"].iloc[0])

print("\nConverting text into numerical features...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["CLEAN_DESCRIPTION"])

y = df["GENRE"]

print("Feature Matrix Shape:", X.shape)
print("Number of Labels:", len(y))

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Create the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predict genres for the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")


# Predict Genre for User Input

while True:

    print("\nEnter a movie description (Type 'exit' to quit):")
    user_input = input("Description: ")

    if user_input.lower() == "exit":
        print("Thank you for using Movie Genre Classifier!")
        break

    # Clean the user's input
    cleaned_input = clean_text(user_input)

    # Convert it to TF-IDF features
    input_vector = vectorizer.transform([cleaned_input])

    # Predict the genre
    prediction = model.predict(input_vector)

    print("\nPredicted Genre:", prediction[0])