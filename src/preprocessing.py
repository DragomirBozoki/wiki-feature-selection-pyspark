from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, expr, arrays_zip, size, lower, regexp_replace,
    concat_ws, split, explode, log2
)

import json
import os

# Load a single JSON wiki file
def load_wiki_json(file_path):
    titles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            if len(article['text']) > 500:
                titles.append(article['title'])
    return titles


# Load all JSON wiki articles from a folder
def load_all_articles(folder_path):
    articles = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith('wiki_'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        article = json.loads(line)
                        articles.append(article)
    return articles


def spark_load_wikisection(path="wikisection_dataset_json/*.json"):
    spark = SparkSession.builder \
        .appName("WikiSection Processing") \
        .master("local[12]") \
        .config("spark.driver.memory", "18g") \
        .config("spark.executor.memory", "18g") \
        .config("spark.executor.cores", "6") \
        .config("spark.driver.maxResultSize", "12g") \
        .config("spark.sql.shuffle.partitions", "96") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    print("✅ Spark session started")

    # Load JSONL (JSON lines) format files
    df = spark.read.option("multiline", "false").json(path)

    print("✅ Loaded WikiSection dataset")
    df.show(5, truncate=500)
    df.printSchema()

    return spark, df

# Load extracted Wikipedia dataset into Spark DataFrame
def spark_load():
    
    spark = SparkSession.builder \
        .appName("WikiExtraction Processing") \
        .master("local[12]") \
        .config("spark.driver.memory", "18g") \
        .config("spark.executor.memory", "18g") \
        .config("spark.executor.cores", "6") \
        .config("spark.driver.maxResultSize", "12g") \
        .config("spark.sql.shuffle.partitions", "96") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()


    print("✅ Spark session started")

    df = spark.read.json("extracted/*")
    df.show(5, truncate=500)
    df.printSchema()

    return spark, df


# Clean text and tokenize
def tokenization(df):
    df_clean = df.withColumn('clean_text', lower(regexp_replace(col('text'), '[^a-zA-Z\s]', '')))
    df_tokens = df_clean.withColumn('unigrams', split(col('clean_text'), '\s+'))
    df_tokens = df_clean.withColumn('tokens', split(col('clean_text'), '\s+'))
    return df_tokens



# Explode tokens and count word frequencies
# Explode tokens and count word frequencies
def word_count(df_tokens):
    df_words = df_tokens.select(explode(col('tokens')).alias('word'))
    word_counts = df_words.groupBy('word').count()
    word_common = word_counts.orderBy('count', ascending=False)
    print("✅ Word counting done")
    return word_counts, word_common


# Calculate total entropy of the vocabulary
def entropy_calc_combined(word_counts):
    

    total_words = word_counts.agg({'count': 'sum'}).collect()[0][0]
    word_probs = word_counts.withColumn('probability', col('count') / total_words)
    word_entropy = word_probs.withColumn('entropy', -col('probability') * log2(col('probability')))
    total_entropy = word_entropy.agg({'entropy': 'sum'}).collect()[0][0]

    print(f"✅ Total entropy of the vocabulary: {total_entropy:.4f} bits")
    return total_entropy


def generate_ngram(df_tokens, n):
    if not isinstance(n, int) or n < 1 or n > 3:
        print('❌ n must be between 1 and 3')
        return df_tokens

    base_col = "tokens"

    # ✅ Filtriraj samo one redove koji imaju dovoljno tokena
    df_tokens = df_tokens.filter(size(col(base_col)) >= n)

    if n == 1:
        # Unigram = samo tokens
        df_tokens = df_tokens.withColumn("unigrams", col(base_col))
    else:
        # ➕ Shiftuj kolone za bigrame/trigrame
        for i in range(1, n):
            df_tokens = df_tokens.withColumn(
                f"tokens_shifted_{i}",
                expr(f"slice({base_col}, {i+1}, size({base_col}) - {i})")
            )

        # 🔗 Zajedničko zipovanje svih n kolona
        shifted_cols = [f"tokens_shifted_{i}" for i in range(1, n)]
        df_tokens = df_tokens.withColumn("token_pairs", arrays_zip(base_col, *shifted_cols))

        # 🧠 Formiraj n-gram stringove
        concat_expr = "transform(token_pairs, x -> concat_ws(' ', " + ', '.join(
            [f"x.tokens_shifted_{i-1}" if i > 1 else f"x.{base_col}" for i in range(1, n+1)]
        ) + "))"

        # 🏷️ Output kolona
        output_col = ["unigrams", "bigrams", "trigrams"][n - 1]
        df_tokens = df_tokens.withColumn(output_col, expr(concat_expr))

    return df_tokens