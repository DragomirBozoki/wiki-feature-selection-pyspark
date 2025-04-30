# Wikipedia Big Data Processing Project

## About
This project processes the full Wikipedia dump to:
- Clean and tokenize the text
- Calculate word entropy and mutual information
- Select the most informative words
- Build feature matrices for text classification
- Train a Logistic Regression classifier

All processing is done using PySpark for scalability.

## Project Structure

data/extracted/ — Raw Wikipedia JSON data

src/preprocessing.py — Text cleaning, tokenization

src/bigdata.py — Entropy calculation, MI, classification

main.py — Main file to run the full pipeline

requirements.txt — Python dependencies

README.md — Project documentation


## Setup
Install the dependencies:
```bash
pip install -r requirements.txt

python main.py

Notes

    Requires ~100GB free space for full Wikipedia dump.

    PySpark driver memory can be adjusted in code if necessary.# wiki-feature-selection-pyspark
