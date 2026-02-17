"""
Intent matching — spaCy-powered semantic similarity with regex fast-path.

Strategy:
  1. Try exact regex match first (fastest, zero ambiguity).
  2. If no regex hit, compute spaCy vector similarity against every
     pattern and pick the best-scoring intent above SIMILARITY_THRESHOLD.
  3. When multiple regex patterns match, prefer the longest (most specific).
"""
import json
import re
import random
import logging

from ..config import INTENTS_PATH

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────
SPACY_MODEL = "en_core_web_md"
SIMILARITY_THRESHOLD = 0.80   # minimum cosine similarity to accept

# ── Normalization helpers (shared by regex path) ──────────
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def _normalize_text(value):
    text = value.lower()
    text = _RE_NON_ALNUM.sub(" ", text)
    return _RE_MULTI_SPACE.sub(" ", text).strip()


# ── Load intents from JSON ────────────────────────────────
def _load_intents():
    try:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("intents", [])
    except FileNotFoundError:
        logger.warning("intents.json not found; intent matching disabled")
        return []
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse intents.json: {exc}")
        return []


INTENTS = _load_intents()
logger.info(f"Loaded {len(INTENTS)} intents")


# ── Load spaCy model ─────────────────────────────────────
_nlp = None
_spacy_failed = False

def _load_spacy():
    global _nlp, _spacy_failed
    if _nlp is not None:
        return _nlp
    if _spacy_failed:
        return None
    try:
        import spacy
        _nlp = spacy.load(SPACY_MODEL)
        logger.info(f"spaCy model '{SPACY_MODEL}' loaded successfully")
        return _nlp
    except Exception as exc:
        _spacy_failed = True
        logger.warning(f"spaCy not available ({exc}) — using regex-only matching")
        return None


# ── Build compiled data structures ────────────────────────
_COMPILED_INTENTS = []   # regex-based
_SPACY_INTENTS = []      # vector-based

for _intent in INTENTS:
    _patterns = _intent.get("patterns", [])
    _responses = _intent.get("responses", [])
    if not _patterns or not _responses:
        continue

    tag = _intent.get("tag", "unknown")

    # Regex patterns
    _compiled = []
    for _p in _patterns:
        _norm = _normalize_text(_p)
        if _norm:
            _compiled.append(re.compile(r"\b" + re.escape(_norm) + r"\b"))
    if _compiled:
        _COMPILED_INTENTS.append({
            "tag": tag,
            "responses": _responses,
            "patterns": _compiled,
        })

    # Raw pattern strings for spaCy similarity (processed lazily)
    _SPACY_INTENTS.append({
        "tag": tag,
        "responses": _responses,
        "raw_patterns": _patterns,
        "pattern_docs": None,      # will be filled on first use
    })

logger.info(f"Precompiled {len(_COMPILED_INTENTS)} intent patterns (regex)")
logger.info(f"Prepared {len(_SPACY_INTENTS)} intents for spaCy similarity")


# ── Lazy init of spaCy pattern docs ──────────────────────
_spacy_ready = False

def _ensure_spacy_patterns():
    """Process all pattern strings into spaCy Doc objects (one-time cost)."""
    global _spacy_ready
    if _spacy_ready:
        return True

    nlp = _load_spacy()
    if nlp is None:
        return False

    for intent in _SPACY_INTENTS:
        if intent["pattern_docs"] is None:
            intent["pattern_docs"] = [nlp(p) for p in intent["raw_patterns"]]

    _spacy_ready = True
    logger.info("spaCy pattern docs initialized")
    return True


# ── Matching engines ──────────────────────────────────────
def _regex_match(normalized):
    """Exact substring match via regex. Returns (tag, responses) or None."""
    best = None   # (match_length, intent_dict)
    for intent in _COMPILED_INTENTS:
        for pat in intent["patterns"]:
            m = pat.search(normalized)
            if m:
                length = m.end() - m.start()
                if best is None or length > best[0]:
                    best = (length, intent)
                break
    return best[1] if best else None


def _spacy_match(message):
    """Semantic similarity match via spaCy word vectors."""
    if not _ensure_spacy_patterns():
        return None

    nlp = _nlp
    msg_doc = nlp(message)

    # Skip if the message has no vector (e.g., only punctuation)
    if not msg_doc.has_vector or msg_doc.vector_norm == 0:
        return None

    best_score = 0.0
    best_intent = None

    for intent in _SPACY_INTENTS:
        for pat_doc in intent["pattern_docs"]:
            if pat_doc.vector_norm == 0:
                continue
            score = msg_doc.similarity(pat_doc)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_intent and best_score >= SIMILARITY_THRESHOLD:
        logger.info(
            f"spaCy match: '{message[:50]}' → {best_intent['tag']} "
            f"(score={best_score:.3f})"
        )
        return best_intent

    logger.debug(
        f"spaCy no match: '{message[:50]}' "
        f"(best={best_score:.3f}, threshold={SIMILARITY_THRESHOLD})"
    )
    return None


# ── Public API ────────────────────────────────────────────
def match_intent(message):
    """
    Return {'tag': ..., 'response': ...} or None.

    Pipeline:
      1. Regex exact match (fast, deterministic)
      2. spaCy semantic similarity (handles synonyms & rephrasings)
    """
    if not _COMPILED_INTENTS and not _SPACY_INTENTS:
        return None

    # --- Pass 1: regex ---
    normalized = _normalize_text(message)
    result = _regex_match(normalized)
    if result:
        return {
            "tag": result["tag"],
            "response": random.choice(result["responses"]),
        }

    # --- Pass 2: spaCy similarity ---
    result = _spacy_match(message)
    if result:
        return {
            "tag": result["tag"],
            "response": random.choice(result["responses"]),
        }

    return None
