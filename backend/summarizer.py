import re
from heapq import nlargest
from string import punctuation

import nltk
import numpy as np
import spacy
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tree import Tree
import transformers as _transformers  # noqa: F401 - optional, only used if transformer models are requested
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
    "NORP ": 0.8
}


# CLEAN TEXT

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_notes(text):
    """Convert a sentence summary into plain-text notes for copy/download."""
    sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text.strip())
    return "\n".join(f"- {sentence.strip()}" for sentence in sentences if sentence.strip())


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

    # NORMALIZE WORD FREQUENCY
    
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


class TransformerModelError(RuntimeError):
    """Raised when a transformer model/tokenizer fails to load or run."""

def get_transformer_status():
    """Return quick diagnostic info about the transformers package and runtime environment."""
    info = {
        "installed": False,
        "module": None,
        "version": None,
        "torch_installed": False,
        "error": None,
    }

    try:
        import transformers
        info["installed"] = True
        info["module"] = getattr(transformers, "__file__", None)
        info["version"] = getattr(transformers, "__version__", None)
    except Exception as exc:  # pragma: no cover - diagnostic only
        info["error"] = str(exc)
        return info

    try:
        import torch
        info["torch_installed"] = True
    except Exception:  # pragma: no cover - diagnostic only
        pass

    return info

_TRANSFORMER_MODEL_CACHE = {}


def _load_transformer_cached(name, loader):
    if name not in _TRANSFORMER_MODEL_CACHE:
        try:
            _TRANSFORMER_MODEL_CACHE[name] = loader()
        except Exception as exc:  # noqa: BLE001 - surface as one clear error type
            raise TransformerModelError(
                f"Could not load transformer model '{name}': {exc}"
            ) from exc
    return _TRANSFORMER_MODEL_CACHE[name]


def preload_transformer_models(models=("bert", "t5", "gpt2")):
    """
    Optional: call at startup (like initialize_nlp_engine() does for
    the NLTK/spaCy pipeline) to download/load weights up front
    instead of on the first request.
    """
    loaders = {"bert": _get_bert, "t5": _get_t5, "gpt2": _get_gpt2}
    for name in models:
        loaders[name]()


# ---------- BERT: extractive, via sentence embeddings ----------

def _get_bert():
    def load():
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        model.eval()
        return tokenizer, model

    return _load_transformer_cached("bert", load)


def _bert_sentence_embeddings(sentences, tokenizer, model):
    import torch

    embeddings = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(
                sentence, return_tensors="pt", truncation=True, max_length=512
            )
            outputs = model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(vector)
    return np.array(embeddings)


def bert_extractive_summarize(text, num_sentences=3):
    """
    Extractive summary using BERT embeddings + KMeans clustering
    (one representative sentence per cluster), restored to
    original order. A second, embeddings-based way to pick
    sentences - separate from the NLTK/spaCy scoring in Part 1.
    """
    from sklearn.cluster import KMeans

    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    tokenizer, model = _get_bert()
    embeddings = _bert_sentence_embeddings(sentences, tokenizer, model)

    n_clusters = min(num_sentences, len(sentences))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings)

    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_indices = [
            i for i, label in enumerate(kmeans.labels_) if label == cluster_id
        ]
        centroid = kmeans.cluster_centers_[cluster_id]
        closest = min(
            cluster_indices, key=lambda i: np.linalg.norm(embeddings[i] - centroid)
        )
        selected_indices.append(closest)

    selected_indices.sort()
    return " ".join(sentences[i] for i in selected_indices)


# ---------- T5: abstractive ----------

def _get_t5():
    def load():
        from transformers import pipeline

        return pipeline("summarization", model="t5-small", tokenizer="t5-small")

    return _load_transformer_cached("t5", load)


def t5_summarize(text, max_length=150, min_length=30, output_format="text", user_request=""):
    summarize_pipe = _get_t5()
    request_hint = f" Follow this request: {user_request}." if user_request else ""
    format_hint = " Write the result as bullet notes." if output_format == "notes" else ""
    prefixed_text = "summarize:" + request_hint + format_hint + " " + text
    input_words = len(text.split())
    # Never request more output than the source can reasonably support, and
    # always leave enough room for a complete result at each preset.
    max_length = min(max_length, max(40, input_words))
    min_length = min(min_length, max(20, max_length - 10))
    result = summarize_pipe(
        prefixed_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]


# ---------- GPT-2: abstractive via prompting (experimental) ----------

def _get_gpt2():
    def load():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        return tokenizer, model

    return _load_transformer_cached("gpt2", load)


def gpt2_summarize(text, max_new_tokens=60, length="balanced", output_format="text", user_request=""):
    import torch

    tokenizer, model = _get_gpt2()
    request_hint = f" Follow this request: {user_request}." if user_request else ""
    format_hint = " Use one bullet point per idea." if output_format == "notes" else ""
    prompt = (
        "Write a concise, factual summary of the following text."
        + request_hint
        + format_hint
        + " Keep the important names, dates, facts, and conclusions.\n\n"
        + text.strip()
        + "\n\nSummary:"
    )

    # GPT-2 has a 1,024-token context window. Reserve room for the answer
    # instead of allowing a long input to make generation fail or truncate.
    context_limit = min(getattr(model.config, "n_positions", 1024), 1024)
    answer_tokens = min(max_new_tokens, max(32, context_limit // 5))
    input_limit = context_limit - answer_tokens
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=input_limit,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=answer_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    generated = generated.split("\n")[0].strip()

    # Base GPT-2 sometimes repeats the prompt or produces no useful answer.
    if len(generated.split()) < 5 or generated.lower().startswith("summary:"):
        return _gpt2_fallback_summary(text, length, output_format)
    return generated


def _gpt2_fallback_summary(text, length="balanced", output_format="text"):
    """Return a complete, faithful result when GPT-2 generates unusable text."""
    sentences = sent_tokenize(clean_text(text))
    if not sentences:
        return ""

    ratios = {"short": 0.2, "balanced": 0.4, "detailed": 0.7}
    limit = max(1, min(len(sentences), round(len(sentences) * ratios.get(length, 0.4))))
    summary = " ".join(sentences[:limit])
    return to_notes(summary) if output_format == "notes" else summary


# ---------- unified entry point ----------

SUPPORTED_TRANSFORMER_MODELS = {"bert", "t5", "gpt2"}

TRANSFORMER_LENGTHS = {
    "short": {"bert": 0.25, "t5": (90, 20), "gpt2": 60},
    "balanced": {"bert": 0.45, "t5": (160, 35), "gpt2": 100},
    "detailed": {"bert": 0.70, "t5": (240, 60), "gpt2": 160},
}


def transformer_summarize(text, model="t5", **kwargs):
    """
    Single call site for the API layer. `model` is one of:
      "bert" (extractive), "t5" (abstractive), "gpt2" (abstractive).
    Returns a dict shaped like `summarizer()`'s output (summary +
    error) so both pipelines can be handled the same way downstream.
    """
    if not text or not text.strip():
        return {"summary": "", "model": model, "error": "Please provide some text."}

    length = kwargs.pop("length", "balanced")
    output_format = kwargs.pop("output_format", "text")
    user_request = kwargs.pop("user_request", "")

    if model not in SUPPORTED_TRANSFORMER_MODELS:
        return {
            "summary": "",
            "model": model,
            "error": f"Unknown model '{model}'. Choose from {sorted(SUPPORTED_TRANSFORMER_MODELS)}.",
        }

    if length not in TRANSFORMER_LENGTHS:
        return {"summary": "", "model": model, "length": length, "error": "Unknown summary length."}

    settings = TRANSFORMER_LENGTHS[length][model]

    try:
        if model == "bert":
            sentence_count = len(sent_tokenize(clean_text(text)))
            summary = bert_extractive_summarize(
                text,
                num_sentences=max(1, min(sentence_count, round(sentence_count * settings))),
            )
        elif model == "t5":
            summary = t5_summarize(
                text,
                max_length=settings[0],
                min_length=settings[1],
                output_format=output_format,
                user_request=user_request,
            )
        else:
            summary = gpt2_summarize(
                text,
                max_new_tokens=settings,
                length=length,
                output_format=output_format,
                user_request=user_request,
            )
        return {
            "summary": summary,
            "model": model,
            "length": length,
            "output_format": output_format,
            "analysis": {
                "type": "transformer",
                "model": model,
                "length": length,
                "output_format": output_format,
                "user_request": user_request or None,
            },
            "error": None,
        }
    except (TransformerModelError, ModuleNotFoundError, ImportError, OSError, RuntimeError) as exc:
        return {
            "summary": "",
            "model": model,
            "length": length,
            "output_format": output_format,
            "error": f"{model.upper()} is unavailable: {exc}",
        }