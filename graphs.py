import os
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def load_top_words(file_path, top_n=20):
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f.readlines()]
    return words[:top_n]

def plot_accuracy_bar(unigram_acc, bigram_acc, tfidf_acc):
    methods = ["Unigram", "Bigram", "TF-IDF"]
    accuracies = [unigram_acc, bigram_acc, tfidf_acc]
    plt.figure(figsize=(7, 5))
    plt.bar(methods, accuracies)
    plt.title("Accuracy Comparison of Methods")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_wordcloud(words, title):
    text = " ".join(words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_mi_words(unigrams, bigrams):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    axes[0].barh(unigrams[::-1], range(1, len(unigrams) + 1))
    axes[0].set_title("Top Informative Unigrams (by MI)")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Unigram")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    axes[1].barh(bigrams[::-1], range(1, len(bigrams) + 1))
    axes[1].set_title("Top Informative Bigrams (by MI)")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Bigram")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load results
    with open("results/classification_results.json") as f:
        classification_results = json.load(f)
    with open("results/tfidf_results.json") as f:
        tfidf_results = json.load(f)

    unigram_acc = classification_results["unigram_accuracy"]
    bigram_acc = classification_results["bigram_accuracy"]
    tfidf_acc = tfidf_results["tfidf_accuracy"]

    # Plot accuracy comparison
    plot_accuracy_bar(unigram_acc, bigram_acc, tfidf_acc)

    # Load top words
    top_unigrams = load_top_words("results/list_MI_words_unigrams.txt")
    top_bigrams = load_top_words("results/list_MI_words_bigrams.txt")

    # Plot MI bar charts
    plot_mi_words(top_unigrams, top_bigrams)

    # Plot word clouds
    plot_wordcloud(top_unigrams, "Word Cloud of Top MI Unigrams")
    plot_wordcloud(top_bigrams, "Word Cloud of Top MI Bigrams")

    # Analyze intersection
    intersect = set(top_unigrams) & set([b.split('_')[0] for b in top_bigrams])
    print("\n🔍 Overlapping informative tokens between unigrams and bigrams:")
    print(sorted(intersect) if intersect else "(none)")