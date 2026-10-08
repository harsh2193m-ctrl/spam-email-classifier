import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "spam_messages.csv"

df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["label", "message"])

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

print("\nTry your own message.")
message = input("Enter message: ")
result = model.predict([message])[0]
print("Prediction:", result)
