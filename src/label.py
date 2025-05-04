def label_text(text):
    text = text.lower()

    topics = {
        "science": ["physics", "biology", "chemistry", "astronomy", "laboratory", "experiment", "genetics", "theory", "research", "scientist",
                    "neuron", "molecule", "atom", "quantum", "evolution", "ecology", "zoology", "botany", "biology", "science"],
        
        "technology": ["software", "hardware", "computer", "algorithm", "machine learning", "ai", "cyber", "cloud", "programming", "robotics",
                       "data", "network", "mobile", "android", "python", "code", "developer", "technology", "virtual", "digital"],
        
        "politics": ["election", "government", "senate", "parliament", "president", "minister", "law", "constitution", "democracy", "political",
                     "vote", "party", "policy", "diplomacy", "campaign", "republic", "congress", "governor", "legislation", "referendum"],
        
        "sports": ["football", "soccer", "basketball", "olympics", "tennis", "athlete", "match", "tournament", "league", "goal",
                   "team", "coach", "player", "medal", "competition", "training", "race", "championship", "score", "stadium"],
        
        "health": ["doctor", "hospital", "medicine", "disease", "vaccine", "surgery", "pandemic", "virus", "infection", "treatment",
                   "mental", "nutrition", "wellness", "fitness", "clinic", "diagnosis", "symptom", "pharmacy", "public health", "covid"],
        
        "business": ["finance", "economy", "market", "bank", "stock", "investment", "trade", "company", "startup", "profit",
                     "capital", "entrepreneur", "share", "corporation", "industry", "revenue", "inflation", "budget", "tax", "commerce"],
        
        "entertainment": ["movie", "cinema", "music", "concert", "actor", "actress", "tv", "show", "album", "netflix",
                          "drama", "series", "theater", "director", "hollywood", "celebrity", "comedy", "scene", "song", "performance"],
        
        "history": ["empire", "revolution", "ancient", "king", "queen", "dynasty", "colonial", "historian", "medieval", "war",
                    "treaty", "era", "civilization", "artifact", "regime", "timeline", "conflict", "rebellion", "timeline", "archive"],
        
        "geography": ["continent", "mountain", "river", "island", "ocean", "climate", "city", "country", "valley", "region",
                      "glacier", "volcano", "desert", "peninsula", "lake", "map", "territory", "border", "landscape", "urban"],
        
        "military": ["army", "navy", "soldier", "weapon", "battle", "warfare", "air force", "missile", "military", "strategy",
                     "defense", "tank", "uniform", "general", "combat", "troop", "gun", "conflict", "base", "peacekeeping"],
        
        "religion": ["church", "god", "bible", "christian", "muslim", "islam", "hindu", "prayer", "faith", "religion",
                     "monk", "temple", "mosque", "christ", "belief", "ritual", "holy", "scripture", "saint", "doctrine"],
        
        "philosophy": ["philosopher", "logic", "ethics", "metaphysics", "epistemology", "socrates", "plato", "aristotle", "thought", "reason",
                       "morality", "theory", "existence", "truth", "wisdom", "ideology", "concept", "dialectic", "essence", "being"],
        
        "food": ["fruit", "vegetable", "nut", "cuisine", "dessert", "recipe", "almond", "chocolate", "meat", "fish",
                 "blanched", "spice", "dish", "cooking", "kitchen", "ingredient", "baking", "honey", "marzipan", "sugar"],
    }

    for label, keywords in topics.items():
        if any(keyword in text for keyword in keywords):
            return label

    return "other"
