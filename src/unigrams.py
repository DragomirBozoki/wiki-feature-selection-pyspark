from src.preprocessing import *
from pyspark.sql.functions import col, explode, count, log2, array_contains
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



# Function: Calculate entropy for the top x most frequent words
def entropy_calc_topwords(word_counts, x):
    
    if not isinstance(x, int):
        print('Unesi integer.')
        return
    
    word_sorted = word_counts.orderBy('count', ascending=False).limit(x)
    
    total_words = word_sorted.agg({'count': 'sum'}).collect()[0][0]
    
    word_probs = word_sorted.withColumn('probability', col('count') / total_words)
    word_entropy = word_probs.withColumn('entropy', -col('probability') * log2(col('probability')))
    
    total_entropy_x = word_entropy.agg({'entropy': 'sum'}).collect()[0][0]
    
    print(f"Total entropy of the top {x} words: {total_entropy_x:.4f} bits")
    
    return total_entropy_x


# Function: Prepare data for MI calculation
def prepare_mi_data(df_tokens):

    df_words_labels = df_tokens.select('label', explode(col('tokens')).alias('word'))
    
    word_label_counts = df_words_labels.groupBy('word', 'label').count()
    word_counts = df_words_labels.groupBy('word').count().withColumnRenamed('count', 'word_total')
    label_counts = df_tokens.groupBy('label').count().withColumnRenamed('count', 'label_total')
    
    return word_label_counts, word_counts, label_counts


# Function: Calculate Mutual Information
def calculate_mi(df_tokens):

    word_label_counts, word_counts, label_counts = prepare_mi_data(df_tokens)
    
    word_joined = word_label_counts.join(word_counts, on='word')
    word_joined_label = word_joined.join(label_counts, on='label')
    
    total_docs = label_counts.agg({'label_total': 'sum'}).collect()[0][0]
    
    word_joined_label = word_joined_label.withColumn('p_xy', col('count') / total_docs)
    word_joined_label = word_joined_label.withColumn('p_x', col('word_total') / total_docs)
    word_joined_label = word_joined_label.withColumn('p_y', col('label_total') / total_docs)
    
    word_joined_label = word_joined_label.withColumn(
        'MI',
        log2(col('p_xy') / (col('p_x') * col('p_y')))
    )
    
    word_joined_label.orderBy("MI", ascending=False).show(20, truncate=False)
    
    word_joined_label_limit = word_joined_label.orderBy("MI", ascending=False).limit(500)
    list_MI_words = word_joined_label_limit.select('word').rdd.flatMap(lambda x: x).collect()
    
    for word in list_MI_words:
        df_tokens = df_tokens.withColumn(word, array_contains(col('tokens'), word).cast('integer'))
    
    return df_tokens, list_MI_words
