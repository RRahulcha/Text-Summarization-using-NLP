"""
summarizer.py
=========================================================
Core NLP extractive text summarization module.

This module is a direct refactor of the user's original
`textsummary.py` script. The summarization ALGORITHM is
UNCHANGED — the same NLTK + spaCy hybrid scoring pipeline
(word frequency, POS weighting, NER weighting, sentence
position, sentence length) is preserved exactly.

What changed vs. the original script:
  * The hard-coded sample `text` variable and the
    "run on import" block at the bottom of the original
    file were removed. The API now supplies user text.
  * `ensure_nltk_resources()` and the spaCy model load are
    wrapped so they run once, safely, and raise a clear,
    catchable error instead of crashing the process if the
    spaCy model is missing.
  * Everything else (function names, scoring weights,
    scoring formula, sentence selection logic) is identical
    to the original file.
=========================================================
"""

import re
from heapq import nlargest
from string import punctuation

import nltk
import spacy
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree


# EXCEPTIONS

class SpacyModelMissingError(RuntimeError):
    """Raised when the required spaCy model is not installed."""


# NLTK RESOURCES

def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "chunkers/maxent_ne_chunker": "maxent_ne_chunker",
        "chunkers/maxent_ne_chunker_tab": "maxent_ne_chunker_tab",
        "corpora/words": "words",
    }

    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading missing NLTK resource: {resource_name}")
            if not nltk.download(resource_name, quiet=True):
                raise RuntimeError(
                    f"Unable to download the NLTK resource '{resource_name}'."
                )


# LOAD SPACY (lazy, cached, safe)

_nlp = None
_stopwords_ready = False


def get_spacy_model():
    """
    Load and cache the spaCy `en_core_web_sm` model.
    Raises SpacyModelMissingError with clear install
    instructions if it is not installed, instead of
    letting the raw OSError bubble up to the client.
    """
    global _nlp

    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise SpacyModelMissingError(
                "The spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc

    return _nlp


def initialize_nlp_engine():
    """
    Call once at FastAPI startup. Downloads NLTK resources
    and loads the spaCy model up front so the first user
    request isn't slowed down (or doesn't fail) mid-request.
    """
    global _stopwords_ready

    ensure_nltk_resources()
    get_spacy_model()
    _stopwords_ready = True


# STOPWORDS

def get_stop_words():
    nltk_stopwords = set(stopwords.words("english"))
    spacy_stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
    return nltk_stopwords | spacy_stopwords


# POS WEIGHTS

POS_WEIGHTS = {
    "NN": 1.2,
    "NNS": 1.2,
    "NNP": 1.5,
    "NNPS": 1.5,

    "VB": 1.1,
    "VBD": 1.1,
    "VBG": 1.1,
    "VBN": 1.1,
    "VBP": 1.1,
    "VBZ": 1.1,

    "JJ": 0.8,
    "JJR": 0.8,
    "JJS": 0.8
}


# SPACY POS WEIGHTS

SPACY_POS_WEIGHTS = {
    "PROPN": 1.5,
    "NOUN": 1.2,
    "VERB": 1.1,
    "ADJ": 0.8
}


# NER WEIGHTS

NER_WEIGHTS = {
    "PERSON": 1.5,
    "ORGANIZATION": 1.5,
    "ORG": 1.5,

    "GPE": 1.3,
    "LOCATION": 1.3,
    "LOC": 1.3,

    "DATE": 1.2,
    "TIME": 1.0,

    "EVENT": 1.2,
    "FACILITY": 1.0,
    "PRODUCT": 1.0,
    "WORK_OF_ART": 1.0,
    "LAW": 1.0,
    "NORP": 0.8
}


# CLEAN TEXT

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# NLTK NER EXTRACTION

def extract_nltk_entities(tagged_words):
    entities = []

    tree = ne_chunk(tagged_words, binary=False)

    for subtree in tree:
        if isinstance(subtree, Tree):
            entity_name = " ".join(word for word, tag in subtree.leaves())
            entity_type = subtree.label()

            entities.append({
                "text": entity_name,
                "label": entity_type
            })

    return entities


# MAIN SUMMARIZER

def summarizer(text, summary_ratio=0.30):
    """
    Extractive text summarizer.

    Pipeline (unchanged from the original textsummary.py):
      1. Clean text
      2. NLTK sentence + word tokenization, POS tagging, NER
      3. spaCy sentence segmentation + NER
      4. Word frequency scoring (normalized)
      5. Hybrid sentence scoring:
         word frequency + NLTK POS + spaCy POS +
         NLTK NER + spaCy NER + position + length
      6. Select top-N sentences by score, restore original
         order, join into the final summary.
    """

    # INPUT VALIDATION
    
    if not text or not text.strip():
        return {
            "summary": "",
            "nltk_sentences": [],
            "spacy_sentences": [],
            "nltk_tokens": [],
            "spacy_tokens": [],
            "nltk_pos": [],
            "spacy_pos": [],
            "nltk_entities": [],
            "spacy_entities": [],
            "sentence_scores": {},
            "error": "Please provide some text."
        }

    nlp = get_spacy_model()
    stop_words = get_stop_words()

    # CLEAN TEXT
    
    text = clean_text(text)

    #                 NLTK PIPELINE
    
    nltk_sentences = sent_tokenize(text)
    nltk_words = word_tokenize(text)
    nltk_tagged_words = pos_tag(nltk_words)
    nltk_entities = extract_nltk_entities(nltk_tagged_words)

    #                 SPACY PIPELINE
    
    doc = nlp(text)
    spacy_sentences = list(doc.sents)

    spacy_entities = []
    for entity in doc.ents:
        spacy_entities.append({
            "text": entity.text,
            "label": entity.label_
        })

    # Guard: extremely short input can produce zero spaCy
    # sentences after cleaning (e.g. only punctuation).
    if not spacy_sentences:
        return {
            "summary": "",
            "nltk_sentences": nltk_sentences,
            "spacy_sentences": [],
            "nltk_tokens": nltk_words,
            "spacy_tokens": [token.text for token in doc],
            "nltk_pos": nltk_tagged_words,
            "spacy_pos": [(token.text, token.pos_) for token in doc],
            "nltk_entities": nltk_entities,
            "spacy_entities": spacy_entities,
            "sentence_scores": {},
            "error": "The text is too short to summarize."
        }

    #        COMBINE NLTK + SPACY INFORMATION
    
    # WORD FREQUENCY
    
    word_frequency = {}

    for word in nltk_words:
        word_lower = word.lower()

        if (
            word_lower not in stop_words
            and word_lower not in punctuation
            and not word.isdigit()
        ):
            word_frequency[word_lower] = word_frequency.get(word_lower, 0) + 1

    # -----------------------------------------------------
    # NORMALIZE WORD FREQUENCY
    # -----------------------------------------------------

    if word_frequency:
        max_frequency = max(word_frequency.values())
        for word in word_frequency:
            word_frequency[word] = word_frequency[word] / max_frequency

    # CREATE NLTK POS LOOKUP

    nltk_pos_lookup = {}
    for index, (word, tag) in enumerate(nltk_tagged_words):
        nltk_pos_lookup[word.lower()] = tag

    # CREATE SPACY NER LOOKUP

    spacy_entity_lookup = {}
    for entity in doc.ents:
        for token in entity:
            spacy_entity_lookup[token.text.lower()] = entity.label_

    # CREATE NLTK NER LOOKUP

    nltk_entity_lookup = {}
    for entity in nltk_entities:
        words = entity["text"].split()
        for word in words:
            nltk_entity_lookup[word.lower()] = entity["label"]

    #        SENTENCE SCORING

    sentence_scores = {}

    for sentence_index, sentence in enumerate(spacy_sentences):
        score = 0.0
        meaningful_words = 0

        # WORD FREQUENCY
        for token in sentence:
            word = token.text.lower()
            if word in word_frequency:
                score += word_frequency[word] * 5.0
                meaningful_words += 1

        # NLTK POS SCORE
        nltk_pos_score = 0.0
        for token in sentence:
            word = token.text.lower()
            if word in nltk_pos_lookup:
                tag = nltk_pos_lookup[word]
                if tag in POS_WEIGHTS:
                    nltk_pos_score += POS_WEIGHTS[tag]

        # SPACY POS SCORE
        spacy_pos_score = 0.0
        for token in sentence:
            if token.pos_ in SPACY_POS_WEIGHTS:
                spacy_pos_score += SPACY_POS_WEIGHTS[token.pos_]

        # SPACY NER SCORE
        spacy_ner_score = 0.0
        for token in sentence:
            word = token.text.lower()
            if word in spacy_entity_lookup:
                entity_type = spacy_entity_lookup[word]
                if entity_type in NER_WEIGHTS:
                    spacy_ner_score += NER_WEIGHTS[entity_type]

        # NLTK NER SCORE
        nltk_ner_score = 0.0
        for token in sentence:
            word = token.text.lower()
            if word in nltk_entity_lookup:
                entity_type = nltk_entity_lookup[word]
                if entity_type in NER_WEIGHTS:
                    nltk_ner_score += NER_WEIGHTS[entity_type]

        # SENTENCE POSITION
        if sentence_index == 0:
            position_score = 1.0
        elif sentence_index == 1:
            position_score = 0.8
        elif sentence_index == len(spacy_sentences) - 1:
            position_score = 0.7
        else:
            position_score = 0.3

        # SENTENCE LENGTH
        sentence_words = [
            token for token in sentence
            if not token.is_punct and not token.is_space
        ]
        sentence_length = len(sentence_words)

        if sentence_length < 5:
            length_score = 0.3
        elif sentence_length > 60:
            length_score = 0.5
        else:
            length_score = 1.0

        # FINAL HYBRID SCORE
        final_score = (
            score
            + (nltk_pos_score * 0.7)
            + (spacy_pos_score * 0.7)
            + (nltk_ner_score * 0.8)
            + (spacy_ner_score * 1.0)
            + (position_score * 1.0)
            + (length_score * 0.5)
        )

        sentence_scores[sentence] = final_score

    # SUMMARY SIZE
    

    select_length = int(len(spacy_sentences) * summary_ratio)
    select_length = max(1, select_length)
    select_length = min(select_length, len(spacy_sentences))

    # SELECT IMPORTANT SENTENCES
    
    selected_sentences = nlargest(
        select_length,
        sentence_scores,
        key=sentence_scores.get
    )

    # RESTORE ORIGINAL ORDER

    selected_sentences = sorted(
        selected_sentences,
        key=lambda sentence: sentence.start
    )

    # FINAL SUMMARY

    summary = " ".join(
        sentence.text.strip() for sentence in selected_sentences
    )

    # RETURN ALL NLP INFORMATION

    return {
        "summary": summary,
        "nltk_sentences": nltk_sentences,
        "spacy_sentences": [sentence.text for sentence in spacy_sentences],
        "nltk_tokens": nltk_words,
        "spacy_tokens": [token.text for token in doc],
        "nltk_pos": nltk_tagged_words,
        "spacy_pos": [(token.text, token.pos_) for token in doc],
        "nltk_entities": nltk_entities,
        "spacy_entities": spacy_entities,
        "sentence_scores": {
            sentence.text: score
            for sentence, score in sentence_scores.items()
        },
        "error": None
    }
