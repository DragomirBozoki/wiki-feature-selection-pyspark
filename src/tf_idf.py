from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import os

def tfidf_classification(df_tokens):
    print(">>> TF-IDF klasifikacija...")

    # Pretvori Spark DataFrame u pandas
    pdf = df_tokens.select("clean_text", "label").dropna().toPandas()

    X = pdf["clean_text"]
    y = pdf["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Sačuvaj model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/tfidf_model.pkl")

    # Sačuvaj metriku
    os.makedirs("results", exist_ok=True)
    with open("results/tfidf_results.json", "w") as f:
        json.dump({"tfidf_accuracy": acc}, f, indent=4)

    print(f"✅ TF-IDF tačnost: {acc:.4f}")
    print("💾 Model sačuvan u models/tfidf_model.pkl")
    
    return pipeline, acc