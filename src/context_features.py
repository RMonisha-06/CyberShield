import re


# Small rule-based lexicons for demo-level sentiment cues.
# These are descriptive context features, not the classifier itself.

NEGATIVE_WORDS = {
    "hate", "stupid", "idiot", "loser", "ugly",
    "dumb", "shut", "worst", "kill", "die",
    "annoying", "pathetic", "disgusting", "worthless"
}

POSITIVE_WORDS = {
    "good", "great", "nice", "love", "happy",
    "awesome", "excellent", "thanks", "thank",
    "beautiful", "amazing"
}


def extract_context_features(text):
    """
    Extract interpretable contextual/linguistic features
    for display in the CyberShield demo.

    These features are NOT currently fused into the
    trained Bi-LSTM classifier.
    """

    text = text or ""

    words = re.findall(r"\b\w+\b", text)

    word_count = len(words)
    character_count = len(text)

    # -----------------------------
    # Emoji detection
    # -----------------------------

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+"
    )

    emojis = emoji_pattern.findall(text)

    emoji_detected = len(emojis) > 0

    # -----------------------------
    # Capitalization
    # -----------------------------

    letters = [c for c in text if c.isalpha()]

    uppercase_count = sum(
        1 for c in letters if c.isupper()
    )

    uppercase_ratio = (
        uppercase_count / len(letters)
        if letters
        else 0
    )

    if uppercase_ratio >= 0.7 and len(letters) >= 3:
        capitalization = "High capitalization"
    elif uppercase_ratio > 0:
        capitalization = "Mixed capitalization"
    else:
        capitalization = "Normal"

    # -----------------------------
    # Repeated punctuation
    # -----------------------------

    repeated_punctuation = bool(
        re.search(r"[!?.,]{2,}", text)
    )

    # -----------------------------
    # Exclamation / question cues
    # -----------------------------

    exclamation_count = text.count("!")
    question_count = text.count("?")

    # -----------------------------
    # Sentiment cue estimation
    # -----------------------------

    lower_words = {
        word.lower()
        for word in words
    }

    negative_hits = sorted(
        lower_words.intersection(NEGATIVE_WORDS)
    )

    positive_hits = sorted(
        lower_words.intersection(POSITIVE_WORDS)
    )

    if len(negative_hits) > len(positive_hits):
        sentiment = "Negative"
    elif len(positive_hits) > len(negative_hits):
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    # -----------------------------
    # Potentially aggressive cues
    # -----------------------------

    aggressive_cues = bool(
        negative_hits
        or exclamation_count >= 2
        or uppercase_ratio >= 0.7
    )

    # -----------------------------
    # Return structured features
    # -----------------------------

    return {
        "word_count": word_count,
        "character_count": character_count,

        "emoji_detected": emoji_detected,
        "emoji_count": len(emojis),
        "emojis": emojis,

        "sentiment": sentiment,

        "negative_cues": negative_hits,
        "positive_cues": positive_hits,

        "repeated_punctuation": repeated_punctuation,

        "exclamation_count": exclamation_count,
        "question_count": question_count,

        "capitalization": capitalization,
        "uppercase_ratio": round(
            uppercase_ratio,
            2
        ),

        "aggressive_cues": aggressive_cues
    }