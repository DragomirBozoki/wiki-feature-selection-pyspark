import re
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def label_text(text):
    text = text.lower()

    # Izdvajanje reči i lematizacija
    raw_words = re.findall(r'\b\w+\b', text)
    words = set(lemmatizer.lemmatize(word) for word in raw_words)

    # Tematski rečnik
    topics = {
        "science": [
            "physics", "biology", "chemistry", "astronomy", "genetics", "experiment", "theory",
            "laboratory", "scientific", "evolution", "neuroscience", "molecule", "atom", "cell",
            "research", "discovery", "scientist", "hypothesis", "biochemistry", "universe", "data",
            "analysis", "ecosystem", "biology", "microscope"
        ],
        "technology": [
            "technology", "software", "hardware", "computer", "ai", "robotics", "automation",
            "cybersecurity", "network", "database", "cloud", "algorithm", "coding", "programming",
            "machine", "artificial", "intelligence", "internet", "blockchain", "gadget",
            "smartphone", "iot", "web", "server", "code"
        ],
        "politics": [
            "president", "government", "minister", "law", "policy", "parliament", "congress", "senate",
            "democracy", "election", "campaign", "political", "diplomacy", "regulation", "constitution",
            "governor", "legislation", "republic", "vote", "embassy", "authority", "reform", "party",
            "coalition", "administration"
        ],
        "sports": [
            "football", "soccer", "basketball", "tennis", "olympics", "athlete", "match", "goal",
            "coach", "tournament", "championship", "league", "win", "team", "game", "player",
            "referee", "stadium", "ranking", "medal", "volleyball", "score", "racing", "cricket",
            "fifa"
        ],
        "health": [
            "hospital", "doctor", "nurse", "medicine", "healthcare", "covid", "pandemic", "vaccine",
            "infection", "disease", "virus", "treatment", "mental", "illness", "epidemic",
            "surgery", "clinic", "symptom", "pharmaceutical", "public", "nutrition", "therapy",
            "diagnosis", "immunity", "antibiotic"
        ],
        "business": [
            "economy", "finance", "market", "stock", "investment", "trade", "company", "startup",
            "entrepreneur", "bank", "revenue", "profit", "corporation", "industry", "merger",
            "acquisition", "capital", "inflation", "tax", "budget", "shareholder", "dividend",
            "debt", "valuation", "income"
        ],
        "entertainment": [
            "movie", "film", "actor", "actress", "cinema", "netflix", "series", "tv", "show",
            "theatre", "celebrity", "music", "album", "song", "concert", "performance", "hollywood",
            "festival", "comedy", "drama", "documentary", "animation", "screenplay", "trailer", "award"
        ],
        "history": [
            "history", "war", "empire", "revolution", "ancient", "medieval", "colonial", "dynasty",
            "battle", "monarchy", "king", "queen", "historian", "timeline", "archaeology", "crusade",
            "civilization", "invasion", "treaty", "historic", "era", "artifact", "manuscript",
            "conquest", "kingdom"
        ],
        "geography": [
            "continent", "country", "city", "river", "mountain", "ocean", "desert", "island",
            "valley", "lake", "climate", "region", "border", "territory", "latitude", "longitude",
            "map", "capital", "population", "glacier", "earthquake", "volcano", "terrain", "forest",
            "plain"
        ],
        "military": [
            "army", "navy", "air", "soldier", "weapon", "missile", "tank", "battle", "military",
            "warfare", "strategy", "combat", "troop", "defense", "marine", "brigade", "base",
            "artillery", "airstrike", "sniper", "operation", "drone", "surveillance", "bomb", "general"
        ],
        "education": [
            "school", "student", "university", "teacher", "education", "class", "lecture",
            "curriculum", "degree", "academic", "exam", "homework", "research", "textbook",
            "scholarship", "grade", "college", "faculty", "campus", "graduate", "thesis",
            "subject", "professor", "institution", "learning"
        ],
        "religion": [
            "church", "god", "faith", "bible", "prayer", "islam", "christian", "muslim", "religion",
            "temple", "spiritual", "jesus", "mosque", "worship", "scripture", "hindu", "jewish",
            "ritual", "holy", "divine", "sacred", "religious", "belief", "pilgrimage", "prophet"
        ],
        "environment": [
            "climate", "pollution", "carbon", "emission", "sustainability", "environment", "recycle",
            "renewable", "solar", "greenhouse", "deforestation", "biodiversity", "ecosystem", "wildlife",
            "conservation", "nature", "organic", "waste", "plastic", "ozone", "energy", "earth",
            "green", "ecology", "habitat"
        ]
    }

    match_counts = Counter()

    for label, keywords in topics.items():
        match_counts[label] = len(words.intersection(keywords))

    best_match = match_counts.most_common(1)

    if best_match and best_match[0][1] > 0:
        return best_match[0][0]

    return "other"
