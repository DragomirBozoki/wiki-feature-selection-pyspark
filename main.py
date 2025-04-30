from src.preprocessing import *
from src.unigrams import *
from src.n_grams import *
from src.classification import classification

# Load data
df = spark_load()

# Tokenization
df_tokens = tokenization(df)

# Unigram analysis
word_counts, word_common = word_count(df_tokens)
total_entropy_unigrams = calculate_entr_n_grams(df_tokens, word_counts, n=1)
df_tokens_unigrams, list_MI_words_unigrams = calculate_mi_ngrams(df_tokens, n=1)

print('Done for words')

# Bigram analysis
df_tokens_bigrams = generate_ngram(df_tokens, 2)
word_counts_bigrams, _ = word_count(df_tokens_bigrams)
total_entropy_bigrams = calculate_entr_n_grams(df_tokens_bigrams, word_counts_bigrams, n=2)
df_tokens_bigrams_with_features, list_MI_words_bigrams = calculate_mi_ngrams(df_tokens_bigrams, n=2)

print('Done for bigrams')


# Trigram analysis
df_tokens_trigrams = generate_ngram(df_tokens, 3)
word_counts_trigrams, _ = word_count(df_tokens_trigrams)
total_entropy_trigrams = calculate_entr_n_grams(df_tokens_trigrams, word_counts_trigrams, n=3)
df_tokens_trigrams_with_features, list_MI_words_trigrams = calculate_mi_ngrams(df_tokens_trigrams, n=3)

print('Done for trigrams')

model_unigram, acc_unigram = classification(df_tokens_unigrams, list_MI_words_unigrams, "label", n=1)
model_bigram, acc_bigram = classification(df_tokens_bigrams_with_features, list_MI_words_bigrams, "label", n=2)
model_trigram, acc_trigram = classification(df_tokens_trigrams_with_features, list_MI_words_trigrams, "label", n=3)

print('DONE')
