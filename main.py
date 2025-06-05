import os
import json
import glob
import joblib

# Lokalni moduli
from src.unigrams import *
from src.n_grams import *
from src.preprocessing import *
from src.classification import *
from src.load_save import load_list_from_txt, save_list_to_txt
from src.label import label_text
from src.pipeline import *
from src.tf_idf import tfidf_classification
# PySpark
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType


if __name__ == "__main__":

    print("PROCESSING... ....")
    print("Starting the session...")
    spark, df = spark_load()

    # Tokenizacija i labelovanje
    df_tokens = tokenization(df)
    label_udf = udf(label_text, StringType())
    df_tokens = df_tokens.withColumn("label", label_udf(col("text")))

    print("\n---LOAD COMPLETE---\n")

    # Obrada n-grama i treniranje modela
    df_tokens_unigrams, list_MI_words_unigrams, df_tokens_bigrams_with_features, list_MI_words_bigrams = processing_ngrams(spark, df_tokens)

    save_list_to_txt(list_MI_words_unigrams, "results/list_MI_words_unigrams.txt")
    save_list_to_txt(list_MI_words_bigrams, "results/list_MI_words_bigrams.txt")
    print("💾 Saved list_MI_words_* list in 'results/'")

    run_classification(df_tokens_unigrams, list_MI_words_unigrams, df_tokens_bigrams_with_features, list_MI_words_bigrams)
    pipeline, acc = tfidf_classification(df_tokens)

    