import os
import json
import glob
from src.unigrams import *
from src.n_grams import *
from src.preprocessing import *
from src.classification import classification
from src.load_save import load_list_from_txt, save_list_to_txt
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType




def processing_ngrams(spark, df_tokens):
    print(">>> Processing UNIGRAMS...")
    df_tokens_unigrams = generate_ngram(df_tokens, n=1)
    word_counts_unigrams, _ = word_count(df_tokens_unigrams)
    total_entropy_unigrams = calculate_entr_n_grams(df_tokens_unigrams, word_counts_unigrams, n=1)

    df_unigram_path = "data/df_tokens_unigrams.parquet"
    list_unigram_path = "data/list_MI_words_unigrams.txt"

    if len(glob.glob(f"{df_unigram_path}/part-*.parquet")) > 0:
        df_tokens_unigrams = spark.read.parquet(df_unigram_path)
        list_MI_words_unigrams = load_list_from_txt(list_unigram_path)
    else:
        df_tokens_unigrams, list_MI_words_unigrams = calculate_mi_ngrams(df_tokens_unigrams, n=1)
        df_tokens_unigrams.write.mode("overwrite").parquet(df_unigram_path)
        save_list_to_txt(list_MI_words_unigrams, list_unigram_path)

    print("✅ Done for unigrams\n")

    print(">>> Processing BIGRAMS...")
    df_tokens_bigrams = generate_ngram(df_tokens, 2)
    word_counts_bigrams, _ = word_count(df_tokens_bigrams)
    total_entropy_bigrams = calculate_entr_n_grams(df_tokens_bigrams, word_counts_bigrams, n=2)

    df_bigram_path = "data/df_tokens_bigrams.parquet"
    list_bigram_path = "data/list_MI_words_bigrams.txt"

    if len(glob.glob(f"{df_bigram_path}/part-*.parquet")) > 0:
        df_tokens_bigrams_with_features = spark.read.parquet(df_bigram_path)
        list_MI_words_bigrams = load_list_from_txt(list_bigram_path)
    else:
        df_tokens_bigrams_with_features, list_MI_words_bigrams = calculate_mi_ngrams(df_tokens_bigrams, n=2)
        df_tokens_bigrams_with_features.write.mode("overwrite").parquet(df_bigram_path)
        save_list_to_txt(list_MI_words_bigrams, list_bigram_path)

    list_MI_words_bigrams = [c.replace(" ", "_") for c in list_MI_words_bigrams]
    list_MI_words_bigrams = [c for c in list_MI_words_bigrams if c in df_tokens_bigrams_with_features.columns]

    print("✅ Done for bigrams\n")

    return df_tokens_unigrams, list_MI_words_unigrams, df_tokens_bigrams_with_features, list_MI_words_bigrams

def run_classification(df_tokens_unigrams, list_MI_words_unigrams, df_tokens_bigrams_with_features, list_MI_words_bigrams):
    print(">>> Training classifiers...")

    model_uni, acc_uni = classification(df_tokens_unigrams, list_MI_words_unigrams, "label", n=1)
    model_bi, acc_bi = classification(df_tokens_bigrams_with_features, list_MI_words_bigrams, "label", n=2)

    print("✅ CLASSIFICATION COMPLETE")
    print(f"→ Unigram Accuracy:  {acc_uni:.4f}")
    print(f"→ Bigram Accuracy:   {acc_bi:.4f}")

    # 💾 Sačuvaj rezultate u JSON fajl
    os.makedirs("results", exist_ok=True)
    with open("results/classification_results.json", "w") as f:
        json.dump({
            "unigram_accuracy": acc_uni,
            "bigram_accuracy": acc_bi
        }, f, indent=4)

    print("📁 Saved metrics to: results/classification_results.json")

