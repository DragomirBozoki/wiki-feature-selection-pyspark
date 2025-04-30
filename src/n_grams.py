from src.preprocessing import *
from src.unigrams import *

from pyspark.sql.functions import col, explode, count, log2, array_contains
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random


def calculate_entr_n_grams(df_tokens,word_counts, n):

    if not isinstance(n, int):
        print('n must be an integer')
        return

    if n < 1 or n > 3:
        print('n must be between 1 and 3')
        return
    
    if n == 1:

        total_entropy = entropy_calc_combined(word_counts)
        #total_entropy_10000 = entropy_calc_topwords(word_counts, 10000)
        #df_tokens_with_features, list_MI_words = calculate_mi(df_tokens)

        print(f'calculated entropy for words')


    else:
        
        df_tokens = generate_ngram(df_tokens, n)
        df_words = df_tokens.select(explode(col(f"{n}-grams")).alias('ngram'))
        df_words = df_words.groupBy('ngram').count()
        total_ngrams = df_words.agg({'count': 'sum'}).collect()[0][0]
        df_words = df_words.withColumn('probability', col('count') / total_ngrams)
        df_words = df_words.withColumn('entropy', -col('probability') * log2(col('probability')))  
        total_entropy = df_words.agg({'entropy': 'sum'}).collect()[0][0]

        print(f'calculated entropy for {n}-gram')
    
    return total_entropy

def calculate_mi_ngrams(df_tokens, n):

    df_words_labels = df_tokens.select('label', explode(col(f'{n}-grams')).alias(f'ngrams'))
    word_label_counts = df_words_labels.groupBy('ngram', 'label').count()
    word_counts = df_words_labels.groupBy('ngram').count()
    label_counts = df_words_labels.groupBy('label').count()


    joined = word_label_counts.join(word_counts.withColumnRenamed('count', 'ngram_total'), on='ngram')
    joined = joined.join(label_counts.withColumnRenamed('count', 'label_total'), on='label')

    joined = joined.withColumn('p_xy', col('count') / total_docs)
    joined = joined.withColumn('p_x', col('ngram_total') / total_docs)
    joined = joined.withColumn('p_y', col('label_total') / total_docs)


    joined = joined.withColumn('MI', log2(col('p_xy') / (col('p_x') * col('p_y'))))

    word_joined_label_limit = joined.orderBy("MI", ascending=False).limit(500)

    list_MI_ngrams = word_joined_label_limit.select('ngram').rdd.flatMap(lambda x: x).collect()

    for ngram in list_MI_ngrams:
        df_tokens = df_tokens.withColumn(ngram, array_contains(col(f"{n}-grams"), ngram).cast('integer'))

    return df_tokens, list_MI_ngrams