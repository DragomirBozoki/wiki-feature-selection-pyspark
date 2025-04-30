# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, regexp_replace, lower, explode, log2
from pyspark.sql.functions import arrays_zip, expr, concat_ws
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


# Load extracted Wikipedia dataset into Spark DataFrame
def spark_load():

    #adjust the memory size, my LapTop is very weak for this type of processing data so I had to lower down config
    spark = SparkSession.builder \
    .appName("WikiExtraction Processing")\
    .config("spark.driver.memory", "1g")\
    .config("spark.executor.memory", "1g")\
    .config("spark.driver.maxResultSize", "512m")\
    .getOrCreate()


    spark.sparkContext.setLogLevel("WARN")
    df = spark.read.json("extracted/*/*")
    
    df.show(5, truncate=500)
    df.printSchema()

    return df


# Clean text and tokenize
def tokenization(df):
    df_clean = df.withColumn('clean_text', lower(regexp_replace(col('text'), '[^a-zA-Z\s]', '')))
    df_tokens = df_clean.withColumn('tokens', split(col('clean_text'), '\s+'))
    df_tokens.select('title', 'tokens').show(5, truncate=100)
    return df_tokens


# Explode tokens and count word frequencies
def word_count(df_tokens):
    df_words = df_tokens.select(explode(col('tokens')).alias('word'))
    word_counts = df_words.groupBy('word').count()
    word_common = word_counts.orderBy('count', ascending=False)
    #word_common.cache()
    #word_common.show(5, truncate=30)
    print("Done counting words")
    return word_counts, word_common


# Calculate total entropy of the vocabulary
def entropy_calc_combined(word_counts):
    total_words = word_counts.agg({'count': 'sum'}).collect()[0][0]
    word_probs = word_counts.withColumn('probability', col('count') / total_words)
    word_entropy = word_probs.withColumn('entropy', -col('probability') * log2(col('probability')))
    total_entropy = word_entropy.agg({'entropy': 'sum'}).collect()[0][0]
    
    print(f"Total entropy of the vocabulary: {total_entropy:.4f} bits")
    return total_entropy


# Generate ngrams from tokens
def generate_ngram(df_tokens, n):
    
    if not isinstance(n, int):
        print('n must be an integer')
        return

    if n < 1 or n > 3:
        print('n must be between 1 and 3')
        return

    if n == 1:
        df_tokens = df_tokens.withColumn("1-grams", col("tokens"))
    else:
        # Shiftovanje tokena
        for i in range(1, n):
            df_tokens = df_tokens.withColumn(f"tokens_shifted_{i}", expr(f"slice(tokens, {i+1}, size(tokens)-{i})"))

        # Formiranje token pairs
        shifted_cols = [f"tokens_shifted_{i}" for i in range(1, n)]
        df_tokens = df_tokens.withColumn("token_pairs", arrays_zip("tokens", *shifted_cols))

        # Spajanje reči u n-gram string
        df_tokens = df_tokens.withColumn(
            f"{n}-grams",
            expr("transform(token_pairs, x -> concat_ws(' ', " +
                 ', '.join([f"x.tokens_shifted_{i-1}" if i > 1 else "x.tokens" for i in range(1, n+1)]) +
                 "))")
        )
        
    return df_tokens
