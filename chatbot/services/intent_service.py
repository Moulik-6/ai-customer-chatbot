"""
Intent matching — loads intents.json, precompiles regex patterns, matches messages.
"""
import json
import re
import random
import logging
from ..config import INTENTS_PATH

logger = logging.getLogger(__name__)

# ── Normalization regex (compiled once) ───────────────────
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def _normalize_text(value):
    normalized = value.lower()
    normalized = _RE_NON_ALNUM.sub(" ", normalized)
    return _RE_MULTI_SPACE.sub(" ", normalized).strip()


# ── Load & compile intents ────────────────────────────────
def _load_intents():
    try:
        with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('intents', [])
    except FileNotFoundError:
        logger.warning("intents.json not found; intent matching disabled")
        return []
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse intents.json: {exc}")
        return []


INTENTS = _load_intents()
logger.info(f"Loaded {len(INTENTS)} intents")

_COMPILED_INTENTS = []
for _intent in INTENTS:
    _patterns = _intent.get('patterns', [])
    _responses = _intent.get('responses', [])
    if not _patterns or not _responses:
        continue
    _compiled = []
    for _p in _patterns:
        _norm = _normalize_text(_p)
        if _norm:
            _compiled.append(re.compile(r"\b" + re.escape(_norm) + r"\b"))
    if _compiled:
        _COMPILED_INTENTS.append({
            'tag': _intent.get('tag', 'unknown'),
            'responses': _responses,
            'patterns': _compiled,
        })

logger.info(f"Precompiled {len(_COMPILED_INTENTS)} intent patterns")


# ── Public API ────────────────────────────────────────────
def match_intent(message):
    """Return {'tag': ..., 'response': ...} or None."""
    if not _COMPILED_INTENTS:
        return None
    normalized = _normalize_text(message)
    for intent in _COMPILED_INTENTS:
        for pat in intent['patterns']:
            if pat.search(normalized):
                return {
                    'tag': intent['tag'],
                    'response': random.choice(intent['responses']),
                }
    return None
