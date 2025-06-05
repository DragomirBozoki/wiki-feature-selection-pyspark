from src.label import label_text  # ako si ga već ubacio kao modul
from pprint import pprint

def load_top_words(file_path, top_n=10):
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f.readlines()]
    return words[:top_n]



# 1. Učitaj top 10 unigrama
top_unigrams = load_top_words("results/list_MI_words_unigrams.txt", top_n=10)

# 2. Mapiraj svaku reč na klasu
unigram_to_class = {word: label_text(word) for word in top_unigrams}

# 3. Prikaz
print("📊 Top Informative Unigrams and Their Likely Class:")
pprint(unigram_to_class)
