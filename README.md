# 🧠 Wikipedia Topic Classification

A modular pipeline for extracting, cleaning, processing, and classifying Wikipedia articles into predefined topics using **Apache Spark** and optionally **Apache Airflow** for orchestration.

---

## 🔧 Features

- ✅ Extract and parse Wikipedia XML dump using `wikiextractor`
- ✅ Clean and tokenize large-scale text data with PySpark
- ✅ Automatically assign semantic labels using keyword-based lemmatized matching
- ✅ Generate unigrams, bigrams, trigrams for each article
- ✅ Compute entropy and mutual information for informative feature selection
- ✅ Train classification models using Spark ML
- ✅ Orchestrate tasks with Apache Airflow (PythonOperator or BashOperator-based DAGs)

---

## 📁 Project Structure

BigData/
├── extracted/ # Extracted JSON articles from Wikipedia
├── data/ # Cached intermediate data (Parquet, .txt)
├── src/
│ ├── preprocessing.py      # Tokenization, entropy, and word count
│ ├── n_grams.py            # MI calculation and n-gram generation
│ ├── unigrams.py           # Basic word frequency features
│ ├── classification.py     # Classification pipeline
│ ├── label.py              # Text labeling via lemmatized topic detection
│ └── pipeline_steps.py     # Modular step functions for DAG integration
├── main.py                 # Standalone script to run the full pipeline
├── dags/
│ ├── bigdata_pipeline.py   # Airflow DAG using PythonOperators
│ └── bash_pipeline.py      # Simple BashOperator DAG (runs main.py)
├── requirements.txt
└── README.md


---

## 📥 Input

- Wikipedia XML dump:  
  `enwiki-latest-pages-articles.xml`

- Use [`wikiextractor`](https://github.com/attardi/wikiextractor) to convert the XML dump to JSON format.

```bash
wikiextractor enwiki-latest-pages-articles.xml -o extracted --json
```

---
## ⚙️ How to Run
▶️ Manually (without Airflow)

# Activate your environment
source venv-bigdata/bin/activate

# Run the full pipeline
python main.py

---

## 🌀 With Airflow
Option 1: Modular DAG with PythonOperators

# Trigger manually from Airflow UI or CLI
airflow dags trigger bigdata_pipeline_dag

Option 2: Bash DAG (runs main.py)

airflow dags trigger bigdata_wiki_pipeline

Labeled articles based on lemmatized word sets into these domains:

    science, technology, politics, sports, health, business

    entertainment, history, geography, military

    New: education, religion, environment

    Articles that don’t fit: other

---
## 📊 Outputs

    Entropy scores for each n-gram level

    Top 500 Mutual Information features (per n-gram)

    Labeled Spark DataFrames

    Trained classification models

    Accuracy printed for each model (unigram, bigram, trigram)

---
## 📌 Requirements

    Python ≥ 3.8

    Apache Spark

    Apache Airflow (optional)

    nltk (for lemmatization)

    PySpark ML

# Install with:
```bash
pip install -r requirements.txt
```
## 👨‍💻 Author

Dragomir Bozoki
Machine Learning | Big Data pipelines | NLP with Spark

---

## 📜 License

MIT – feel free to use, extend, or fork for your own NLP pipelines.
