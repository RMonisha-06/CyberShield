import re


# =========================================================
# Common words used for conservative obfuscation recovery
# =========================================================

COMMON_WORDS = {
    "idiot",
    "stupid",
    "loser",
    "dumb",
    "ugly",
    "worthless",
    "pathetic",
    "disgusting",
}


# =========================================================
# Leetspeak mapping
# =========================================================

LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
})


# =========================================================
# Text normalization
# =========================================================

def normalize_text(text):
    """
    Conservatively normalize common obfuscated words.

    Handles:
    - Leetspeak
    - Punctuation inserted between characters
    - Repeated characters
    - Whitespace normalization

    Only known target words are recovered.
    """

    text = str(text)

    # -----------------------------------------------------
    # Normalize whitespace
    # -----------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    # -----------------------------------------------------
    # Known punctuation-obfuscated words
    # -----------------------------------------------------

    punctuation_recovery = {
        "idot": "idiot",
        "stupid": "stupid",
        "loser": "loser",
        "dumb": "dumb",
        "ugly": "ugly",
        "worthless": "worthless",
        "pathetic": "pathetic",
        "disgusting": "disgusting",
    }


    # -----------------------------------------------------
    # Tokenize
    # -----------------------------------------------------

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text
    )

    normalized_tokens = []

    i = 0


    while i < len(tokens):

        token = tokens[i]


        # -------------------------------------------------
        # Punctuation inserted between word parts
        #
        # Example:
        # id!ot -> id + ! + ot -> idiot
        # -------------------------------------------------

        if (
            i + 2 < len(tokens)
            and re.fullmatch(
                r"[A-Za-z]+",
                token
            )
            and re.fullmatch(
                r"[!@#$%^&*._-]+",
                tokens[i + 1]
            )
            and re.fullmatch(
                r"[A-Za-z]+",
                tokens[i + 2]
            )
        ):

            left = token
            right = tokens[i + 2]

            candidate = (
                left + right
            ).lower()

            if candidate in punctuation_recovery:

                normalized_tokens.append(
                    punctuation_recovery[candidate]
                )

                i += 3
                continue


        # -------------------------------------------------
        # Leetspeak
        #
        # Example:
        # 1d10t -> idiot
        # -------------------------------------------------

        leet_candidate = token.translate(
            LEET_MAP
        )

        if (
            leet_candidate.lower()
            in COMMON_WORDS
            and any(
                char.isdigit()
                for char in token
            )
        ):

            normalized_tokens.append(
                leet_candidate
            )

            i += 1
            continue


        # -------------------------------------------------
        # Repeated characters
        #
        # Example:
        # stuuupid -> stupid
        # -------------------------------------------------

        if re.search(
            r"(.)\1{2,}",
            token
        ):

            repeated_candidate = re.sub(
                r"(.)\1+",
                r"\1",
                token
            )

            if (
                repeated_candidate.lower()
                in COMMON_WORDS
            ):

                normalized_tokens.append(
                    repeated_candidate
                )

                i += 1
                continue


        # -------------------------------------------------
        # Normal token
        # -------------------------------------------------

        normalized_tokens.append(
            token
        )

        i += 1


    # -----------------------------------------------------
    # Reconstruct text
    # -----------------------------------------------------

    normalized = " ".join(
        normalized_tokens
    )


    # Remove spaces before punctuation
    normalized = re.sub(
        r"\s+([!?.,;:])",
        r"\1",
        normalized
    )


    return normalized


# =========================================================
# Obfuscation analysis
# =========================================================

def analyze_obfuscation(text):
    """
    Detect common obfuscation patterns.

    This analysis is used for visualization in the
    CyberShield interface.
    """

    text = str(text)

    detected = []


    # -----------------------------------------------------
    # Leetspeak
    # -----------------------------------------------------

    leet_pattern = re.compile(
        r"[0134579]"
    )

    if leet_pattern.search(text):

        detected.append({
            "type": "Leetspeak",
            "description":
                "Numbers or symbols used as character substitutions",
            "icon": "🔢"
        })


    # -----------------------------------------------------
    # Punctuation obfuscation
    # -----------------------------------------------------

    punctuation_pattern = re.compile(
        r"[A-Za-z]+[!@#$%^&*._-]+[A-Za-z]+"
    )

    if punctuation_pattern.search(text):

        detected.append({
            "type": "Punctuation Obfuscation",
            "description":
                "Punctuation inserted between characters",
            "icon": "🔣"
        })


    # -----------------------------------------------------
    # Spacing obfuscation
    # -----------------------------------------------------

    spacing_pattern = re.compile(
        r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b"
    )

    if spacing_pattern.search(text):

        detected.append({
            "type": "Spacing Obfuscation",
            "description":
                "Characters separated using spaces",
            "icon": "↔️"
        })


    # -----------------------------------------------------
    # Character repetition
    # -----------------------------------------------------

    repetition_pattern = re.compile(
        r"(.)\1{2,}"
    )

    if repetition_pattern.search(text):

        detected.append({
            "type": "Character Repetition",
            "description":
                "Characters repeated multiple times",
            "icon": "🔁"
        })


    # -----------------------------------------------------
    # Possible deletion
    # -----------------------------------------------------

    normalized = normalize_text(text)

    if (
        normalized.lower() != text.lower()
        and not detected
    ):

        detected.append({
            "type": "Possible Character Deletion",
            "description":
                "Text differs from a recoverable vocabulary form",
            "icon": "✂️"
        })


    return detected