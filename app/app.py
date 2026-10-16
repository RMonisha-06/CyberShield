import os
import sys
import json
import re

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# IMPORTS
# ============================================================

from flask import Flask, render_template, request, jsonify

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

from src.normalize_text import (
    normalize_text,
    analyze_obfuscation
)

from src.context_features import (
    extract_context_features
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "context_bilstm.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tokenizer.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Context Bi-LSTM model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Context Bi-LSTM model loaded.")


# ============================================================
# LOAD CUSTOM TOKENIZER
# ============================================================

with open(
    TOKENIZER_PATH,
    "r",
    encoding="utf-8"
) as f:
    tokenizer_data = json.load(f)


token_to_index = tokenizer_data["token_to_index"]

OOV_INDEX = 1
SEQUENCE_LENGTH = 80


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Tokenization used by the trained model.

    lower
    whitespace normalization
    word/punctuation tokenization
    """

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return re.findall(
        r"\w+|[^\w\s]",
        text
    )


def text_to_sequence(text):
    """
    Convert text into the same integer sequence format
    used during model training.
    """

    tokens = tokenize(text)

    sequence = []

    for token in tokens:

        index = token_to_index.get(
            token,
            OOV_INDEX
        )

        sequence.append(index)

    # Padding
    if len(sequence) < SEQUENCE_LENGTH:

        sequence += [
            0
        ] * (
            SEQUENCE_LENGTH - len(sequence)
        )

    # Truncation
    else:

        sequence = sequence[
            :SEQUENCE_LENGTH
        ]

    return sequence


# ============================================================
# DEMO-LEVEL HOSTILE VOCABULARY
# ============================================================
#
# IMPORTANT:
# This is NOT replacing the trained model.
#
# It is only used to make the interactive demonstration
# reliably identify obvious abusive language.
#
# The trained model prediction is still returned separately.
# ============================================================

HOSTILE_WORDS = {

    "idiot",
    "stupid",
    "loser",
    "dumb",
    "ugly",
    "worthless",
    "useless",
    "pathetic",
    "disgusting",
    "moron",
    "fool",
    "jerk",
    "creep",
    "weirdo",
    "failure",
    "trash",
    "garbage",
    "annoying",

    "fuck",
    "fucking",
    "fucked",
    "fucker",
    "motherfucker",
    "shit",
    "shitty",
    "bullshit",
    "bitch",
    "bastard",
    "asshole",
    "dick",
    "douche",
    "crap",

    "kill",
    "die",
    "hurt",
    "attack",
    "destroy",
    "hate",
    "hating",
    "shut"
}


# ============================================================
# FRIENDLY VOCABULARY
# ============================================================

FRIENDLY_WORDS = {

    "haha",
    "hahaha",
    "lol",
    "thanks",
    "thank",
    "nice",
    "great",
    "awesome",
    "good",
    "cool",
    "love",
    "happy",
    "amazing",
    "welcome",
    "friend",
    "coffee",
    "fun",
    "yay",
    "perfect",
    "sure",
    "okay",
    "ok"
}


FRIENDLY_EMOJIS = {
    "😂",
    "🤣",
    "😆",
    "😄",
    "😁",
    "😊",
    "😜",
    "😝",
    "🙃",
    "❤️",
    "❤",
    "💕",
    "👍",
    "😎",
    "🎉",
    "😁"
}


# ============================================================
# BOUNDARY / HARASSMENT PHRASES
# ============================================================

BOUNDARY_PHRASES = {

    "leave me alone",
    "stop messaging me",
    "stop texting me",
    "please stop",
    "go away",
    "don't talk to me",
    "do not talk to me",
    "stop talking to me",
    "stop contacting me",
    "stop bothering me"
}


# ============================================================
# HOSTILE WORD DETECTION
# ============================================================

def detect_hostile_words(text):

    text = str(text).lower()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    found = []

    for word in words:

        if word in HOSTILE_WORDS:

            found.append(word)

    return sorted(
        list(set(found))
    )


# ============================================================
# FRIENDLY INDICATORS
# ============================================================

def detect_friendly_indicators(text):

    text = str(text).lower()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    found_words = []

    for word in words:

        if word in FRIENDLY_WORDS:

            found_words.append(word)

    found_emojis = []

    for emoji in FRIENDLY_EMOJIS:

        if emoji in text:

            found_emojis.append(
                emoji
            )

    return {
        "words": sorted(
            list(set(found_words))
        ),
        "emojis": sorted(
            list(set(found_emojis))
        )
    }


# ============================================================
# BOUNDARY DETECTION
# ============================================================

def detect_boundary(text):

    text = str(text).lower()

    found = []

    for phrase in BOUNDARY_PHRASES:

        if phrase in text:

            found.append(phrase)

    return sorted(
        list(set(found))
    )


# ============================================================
# CONVERSATION ANALYSIS
# ============================================================

def analyze_conversation(
    previous_messages,
    current_text,
    demo_mode
):
    """
    Analyze previous + current conversation.

    This is the APPLICATION-LEVEL contextual analysis.

    It does not claim that the Bi-LSTM itself consumes
    the complete conversation sequence.
    """

    previous_messages = (
        previous_messages
        if isinstance(previous_messages, list)
        else []
    )

    texts = []

    for message in previous_messages:

        if isinstance(message, dict):

            msg_text = message.get(
                "text",
                ""
            )

        else:

            msg_text = str(message)

        if msg_text.strip():

            texts.append(
                msg_text
            )

    current_text = str(
        current_text or ""
    )

    if current_text.strip():

        texts.append(
            current_text
        )

    combined_text = "\n".join(
        texts
    )

    # --------------------------------------------------------
    # Detect hostile indicators across entire conversation
    # --------------------------------------------------------

    hostile_hits = []

    for text in texts:

        hostile_hits.extend(
            detect_hostile_words(text)
        )

    hostile_hits = sorted(
        list(set(hostile_hits))
    )

    # --------------------------------------------------------
    # Friendly indicators
    # --------------------------------------------------------

    friendly_words = []
    friendly_emojis = []

    for text in texts:

        indicators = detect_friendly_indicators(
            text
        )

        friendly_words.extend(
            indicators["words"]
        )

        friendly_emojis.extend(
            indicators["emojis"]
        )

    friendly_words = sorted(
        list(set(friendly_words))
    )

    friendly_emojis = sorted(
        list(set(friendly_emojis))
    )

    # --------------------------------------------------------
    # Boundary phrases
    # --------------------------------------------------------

    boundaries = []

    for text in texts:

        boundaries.extend(
            detect_boundary(text)
        )

    boundaries = sorted(
        list(set(boundaries))
    )

    # --------------------------------------------------------
    # Count hostile messages
    # --------------------------------------------------------

    hostile_message_count = 0

    for text in texts:

        if detect_hostile_words(text):

            hostile_message_count += 1

    # --------------------------------------------------------
    # Count friendly signals
    # --------------------------------------------------------

    friendly_signal_count = (
        len(friendly_words)
        +
        len(friendly_emojis)
    )

    hostile_signal_count = len(
        hostile_hits
    )

    # --------------------------------------------------------
    # Determine conversation type
    # --------------------------------------------------------

    if hostile_signal_count > 0:

        conversation_has_hostile = True

    else:

        conversation_has_hostile = False

    conversation_is_friendly = (
        friendly_signal_count >= 2
        and hostile_signal_count == 0
    )

    # --------------------------------------------------------
    # MODE-SPECIFIC RISK
    # --------------------------------------------------------

    if demo_mode == "friend":

        # Friend mode is intentionally conservative.
        #
        # A clearly friendly conversation should never
        # show a cyberbullying alert just because the
        # standalone model produces a false positive.

        if conversation_is_friendly:

            risk = "Low"

        elif hostile_signal_count >= 2:

            risk = "High"

        elif hostile_signal_count == 1:

            risk = "Medium"

        else:

            risk = "Low"

    elif demo_mode == "cyberbully":

        if (
            hostile_signal_count >= 1
            or len(boundaries) > 0
        ):

            risk = "High"

        else:

            risk = "Medium"

    else:

        # Manual mode

        if (
            hostile_signal_count >= 2
            or len(boundaries) > 0
        ):

            risk = "High"

        elif hostile_signal_count == 1:

            risk = "Medium"

        elif conversation_is_friendly:

            risk = "Low"

        else:

            risk = "Neutral"

    return {

        "previous_message_count": len(
            previous_messages
        ),

        "total_message_count": len(
            texts
        ),

        "hostile_words": hostile_hits,

        "hostile_message_count":
            hostile_message_count,

        "friendly_words":
            friendly_words,

        "friendly_emojis":
            friendly_emojis,

        "boundary_phrases":
            boundaries,

        "hostile_signal_count":
            hostile_signal_count,

        "friendly_signal_count":
            friendly_signal_count,

        "conversation_is_friendly":
            conversation_is_friendly,

        "conversation_has_hostile":
            conversation_has_hostile,

        "risk":
            risk
    }


# ============================================================
# MODEL PREDICTION
# ============================================================

def run_model_prediction(
    normalized_text
):

    sequence = text_to_sequence(
        normalized_text
    )

    text_input = np.array(
        [sequence],
        dtype=np.int32
    )

    # IMPORTANT:
    # Context features MUST remain exactly 11 dimensions.
    context_info = extract_context_features(
        normalized_text
    )

    context_values = np.array(
        [[
            context_info["word_count"],
            context_info["character_count"],
            int(
                context_info["emoji_detected"]
            ),
            context_info["emoji_count"],
            context_info["uppercase_ratio"],
            int(
                context_info["repeated_punctuation"]
            ),
            context_info["exclamation_count"],
            context_info["question_count"],
            len(
                context_info["negative_cues"]
            ),
            len(
                context_info["positive_cues"]
            ),
            int(
                context_info["aggressive_cues"]
            )
        ]],
        dtype=np.float32
    )

    prediction = model.predict(
        {
            "text_input": text_input,
            "context_input": context_values
        },
        verbose=0
    )

    probability = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    label = (
        "Cyberbullying"
        if probability >= 0.5
        else "Non-Cyberbullying"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1.0 - probability
    )

    return {

        "probability": probability,

        "label": label,

        "confidence": confidence,

        "context_features":
            context_info
    }


# ============================================================
# DEMO DECISION
# ============================================================

def determine_demo_alert(
    demo_mode,
    model_result,
    conversation_result,
    hostile_words
):

    model_probability = model_result[
        "probability"
    ]

    model_label = model_result[
        "label"
    ]

    risk = conversation_result[
        "risk"
    ]

    hostile_count = len(
        hostile_words
    )

    friendly = conversation_result[
        "conversation_is_friendly"
    ]

    # ========================================================
    # FRIEND MODE
    # ========================================================

    if demo_mode == "friend":

        # If conversation is clearly friendly,
        # DO NOT display cyberbullying alert.

        if friendly:

            return {
                "status": "safe",
                "title": "No Cyberbullying Detected",
                "message":
                    "The conversation context is friendly.",
                "reason":
                    "Previous and current messages contain "
                    "friendly conversational signals."
            }

        # Strong hostile evidence should still be detected.

        if (
            hostile_count >= 2
            or risk == "High"
        ):

            return {
                "status": "danger",
                "title": "Cyberbullying Detected",
                "message":
                    "Hostile language was detected in the conversation.",
                "reason":
                    "The conversation contains strong hostile indicators."
            }

        # Otherwise don't scare the user with a model-only
        # false positive.

        return {
            "status": "safe",
            "title": "No Cyberbullying Detected",
            "message":
                "No strong contextual evidence of cyberbullying was found.",
            "reason":
                "The overall conversation context does not indicate bullying."
        }

    # ========================================================
    # CYBERBULLY MODE
    # ========================================================

    if demo_mode == "cyberbully":

        if (
            hostile_count >= 1
            or risk == "High"
            or model_label == "Cyberbullying"
        ):

            return {
                "status": "danger",
                "title": "Cyberbullying Detected",
                "message":
                    "Hostile or abusive language was detected.",
                "reason":
                    "The current message and conversation context "
                    "indicate cyberbullying."
            }

        return {
            "status": "warning",
            "title": "Potentially Harmful Content",
            "message":
                "The message requires further review.",
            "reason":
                "Some contextual signals require attention."
        }

    # ========================================================
    # MANUAL MODE
    # ========================================================

    if demo_mode == "manual":

        if (
            hostile_count >= 1
            or risk == "High"
            or model_label == "Cyberbullying"
        ):

            return {
                "status": "danger",
                "title": "Cyberbullying Detected",
                "message":
                    "The current message or its context "
                    "contains bullying-related signals.",
                "reason":
                    "The message was evaluated together with "
                    "the available contextual information."
            }

        if risk == "Low":

            return {
                "status": "safe",
                "title": "No Cyberbullying Detected",
                "message":
                    "No strong cyberbullying indicators were found.",
                "reason":
                    "The available context is predominantly friendly."
            }

        return {
            "status": "safe",
            "title": "No Cyberbullying Detected",
            "message":
                "No strong cyberbullying indicators were found.",
            "reason":
                "The current message does not contain strong hostile signals."
        }

    # Fallback

    return {
        "status": "safe",
        "title": "No Cyberbullying Detected",
        "message": "No strong indicators were found.",
        "reason": "Default analysis."
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        text = str(
            data.get(
                "text",
                ""
            )
        ).strip()

        previous_messages = data.get(
            "conversation",
            []
        )

        demo_mode = data.get(
            "demo_mode",
            "manual"
        )

        if not text:

            return jsonify({
                "error":
                    "Please enter a message."
            }), 400

        # ====================================================
        # NORMALIZATION
        # ====================================================

        normalized = normalize_text(
            text
        )

        # ====================================================
        # OBFUSCATION ANALYSIS
        # ====================================================

        obfuscation = analyze_obfuscation(
            text
        )

        # ====================================================
        # MODEL
        # ====================================================

        model_result = run_model_prediction(
            normalized
        )

        # ====================================================
        # CONVERSATION
        # ====================================================

        conversation_result = analyze_conversation(
            previous_messages,
            normalized,
            demo_mode
        )

        # ====================================================
        # DEMO HOSTILE WORDS
        # ====================================================

        hostile_words = detect_hostile_words(
            normalized
        )

        # ====================================================
        # FINAL DEMO ALERT
        # ====================================================

        demo_alert = determine_demo_alert(
            demo_mode,
            model_result,
            conversation_result,
            hostile_words
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "original_text": text,

            "normalized_text": normalized,

            "changed_by_normalization":
                text != normalized,

            "obfuscation":
                obfuscation,

            "model_prediction":
                model_result["label"],

            "model_probability":
                round(
                    model_result["probability"],
                    4
                ),

            "model_confidence":
                round(
                    model_result["confidence"] * 100,
                    2
                ),

            "context_features":
                model_result["context_features"],

            "conversation_analysis":
                conversation_result,

            "demo_alert":
                demo_alert,

            "alert_level":
                demo_alert["status"]
        })

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({
            "error":
                str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )