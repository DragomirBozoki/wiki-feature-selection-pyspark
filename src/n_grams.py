from src.preprocessing import *
from src.unigrams import *
from pyspark.sql.functions import lit
from pyspark.sql.functions import col, explode, count, log2, array_contains
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random


def calculate_entr_n_grams(df_tokens, word_counts, n):
    if not isinstance(n, int) or n < 1 or n > 3:
        print('❌ n must be between 1 and 3')
        return None

    if n == 1:
        total_entropy = entropy_calc_combined(word_counts)
        print(f"✅ Calculated entropy for unigrams: {total_entropy:.4f} bits")
        return total_entropy

    # Generate n-grams
    df_tokens = generate_ngram(df_tokens, n)
    col_name = ["unigrams", "bigrams", "trigrams"][n - 1]

    # Explode n-grams into rows
    df_ngrams = df_tokens.select(explode(col(col_name)).alias("ngram"))

    # Count occurrences
    df_counts = df_ngrams.groupBy("ngram").count()

    # Total number of n-grams
    total_ngrams_row = df_counts.agg({'count': 'sum'}).collect()
    if not total_ngrams_row or total_ngrams_row[0][0] is None:
        print(f"⚠️ No {col_name} found.")
        return 0.0

    total_ngrams = total_ngrams_row[0][0]

    # Calculate entropy
    df_probs = df_counts.withColumn("probability", col("count") / total_ngrams)
    df_entropy = df_probs.withColumn("entropy", -col("probability") * log2(col("probability")))
    total_entropy = df_entropy.agg({'entropy': 'sum'}).collect()[0][0]

    print(f"✅ Calculated entropy for {col_name}: {total_entropy:.4f} bits")
    return total_entropy

def calculate_mi_ngrams(df_tokens, n):
    if 'label' not in df_tokens.columns:
        raise ValueError("Kolona 'label' ne postoji u dataframe-u.")

    ngram_col = ["unigrams", "bigrams", "trigrams"][n-1]

    df_words_labels = df_tokens.select("label", explode(col(ngram_col)).alias("ngram"))

    word_label_counts = df_words_labels.groupBy('ngram', 'label').count()
    word_counts = df_words_labels.groupBy('ngram').count()
    label_counts = df_words_labels.groupBy('label').count()

    total_docs = df_tokens.count()

    joined = word_label_counts.join(word_counts.withColumnRenamed('count', 'ngram_total'), on='ngram')
    joined = joined.join(label_counts.withColumnRenamed('count', 'label_total'), on='label')

    joined = joined.withColumn('p_xy', col('count') / total_docs)
    joined = joined.withColumn('p_x', col('ngram_total') / total_docs)
    joined = joined.withColumn('p_y', col('label_total') / total_docs)

    joined = joined.withColumn('MI', log2(col('p_xy') / (col('p_x') * col('p_y'))))

    word_joined_label_limit = joined.orderBy("MI", ascending=False).limit(500)
    list_MI_ngrams = word_joined_label_limit.select('ngram').rdd.flatMap(lambda x: x).collect()

    for ngram in list_MI_ngrams:
        safe_ngram = ngram.replace(" ", "_")
        df_tokens = df_tokens.withColumn(safe_ngram, array_contains(col(ngram_col), ngram).cast('integer'))

    return df_tokens, list_MI_ngrams

