import os

# Lokalni moduli
from src.unigrams import *
from src.n_grams import *
from src.preprocessing import *
from src.classification import *
from src.load_save import load_list_from_txt, save_list_to_txt
from src.label import label_text

# PySpark
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType


# --------------------------------------------
# Start Spark session
# --------------------------------------------

spark, df = spark_load()

# --------------------------------------------
# Tokenizacija i labelovanje
# --------------------------------------------
df_tokens = tokenization(df)
label_udf = udf(label_text, StringType())
df_tokens = df_tokens.withColumn("label", label_udf(col("text")))

print("\n---LOAD COMPLETE---\n")

# --------------------------------------------
# UNIGRAMS
# --------------------------------------------

print(">>> Processing UNIGRAMS...")
word_counts, word_common = word_count(df_tokens)
total_entropy_unigrams = calculate_entr_n_grams(df_tokens, word_counts, n=1)

df_unigram_path = "data/df_tokens_unigrams.parquet"
list_unigram_path = "data/list_MI_words_unigrams.txt"

if os.path.exists(df_unigram_path):
    df_tokens_unigrams = spark.read.parquet(df_unigram_path)
    list_MI_words_unigrams = load_list_from_txt(list_unigram_path)
else:
    df_tokens_unigrams, list_MI_words_unigrams = calculate_mi_ngrams(df_tokens, n=1)
    df_tokens_unigrams.write.mode("overwrite").parquet(df_unigram_path)
    save_list_to_txt(list_MI_words_unigrams, list_unigram_path)

print("✅ Done for unigrams\n")

# --------------------------------------------
# BIGRAMS
# --------------------------------------------
print(">>> Processing BIGRAMS...")
df_tokens_bigrams = generate_ngram(df_tokens, 2)
word_counts_bigrams, _ = word_count(df_tokens_bigrams)
total_entropy_bigrams = calculate_entr_n_grams(df_tokens_bigrams, word_counts_bigrams, n=2)

df_bigram_path = "data/df_tokens_bigrams.parquet"
list_bigram_path = "data/list_MI_words_bigrams.txt"

if os.path.exists(df_bigram_path):
    df_tokens_bigrams_with_features = spark.read.parquet(df_bigram_path)
    list_MI_words_bigrams = load_list_from_txt(list_bigram_path)
else:
    df_tokens_bigrams_with_features, list_MI_words_bigrams = calculate_mi_ngrams(df_tokens_bigrams, n=2)
    df_tokens_bigrams_with_features.write.mode("overwrite").parquet(df_bigram_path)
    save_list_to_txt(list_MI_words_bigrams, list_bigram_path)

print("✅ Done for bigrams\n")

# --------------------------------------------
# TRIGRAMS
# --------------------------------------------
print(">>> Processing TRIGRAMS...")
df_tokens_trigrams = generate_ngram(df_tokens, 3)
word_counts_trigrams, _ = word_count(df_tokens_trigrams)
total_entropy_trigrams = calculate_entr_n_grams(df_tokens_trigrams, word_counts_trigrams, n=3)

df_trigram_path = "data/df_tokens_trigrams.parquet"
list_trigram_path = "data/list_MI_words_trigrams.txt"

if os.path.exists(df_trigram_path):
    df_tokens_trigrams_with_features = spark.read.parquet(df_trigram_path)
    list_MI_words_trigrams = load_list_from_txt(list_trigram_path)
else:
    df_tokens_trigrams_with_features, list_MI_words_trigrams = calculate_mi_ngrams(df_tokens_trigrams, n=3)
    df_tokens_trigrams_with_features.write.mode("overwrite").parquet(df_trigram_path)
    save_list_to_txt(list_MI_words_trigrams, list_trigram_path)

print("✅ Done for trigrams\n")

# --------------------------------------------
# KLASIFIKACIJA
# --------------------------------------------
print(">>> Training classifiers...")
model_unigram, acc_unigram = classification(df_tokens_unigrams, list_MI_words_unigrams, "label", n=1)
model_bigram, acc_bigram = classification(df_tokens_bigrams_with_features, list_MI_words_bigrams, "label", n=2)
model_trigram, acc_trigram = classification(df_tokens_trigrams_with_features, list_MI_words_trigrams, "label", n=3)

print("CLASSIFICATION COMPLETE")
print(f"→ Unigram Accuracy:  {acc_unigram:.4f}")
print(f"→ Bigram Accuracy:   {acc_bigram:.4f}")
print(f"→ Trigram Accuracy:  {acc_trigram:.4f}")
