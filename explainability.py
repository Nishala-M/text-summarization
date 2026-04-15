# explainability.py — Final Submission Version
# ─────────────────────────────────────────────────────────────────────────────
# Scores each source sentence against the generated summary using TF-IDF
# cosine similarity, returning the top N most relevant sentences in reading
# order.  Zone-based selection guarantees coverage across the whole document.
# ─────────────────────────────────────────────────────────────────────────────

import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Compiled patterns ─────────────────────────────────────────────────────────
_RE_ALLCAPS      = re.compile(r"^[A-Z\s\-:]{4,}$")
_RE_INLINE_LABEL = re.compile(
    r"^(abstract|introduction|conclusion[s]?|summary|overview|background|"
    r"methodology|methods|results|discussion|related work|future work)[:\s]+",
    re.IGNORECASE)
_RE_ALLCAPS_PFX  = re.compile(r"^[A-Z][A-Z\s\-\u2013\u2014]{8,}\s+[A-Z][a-z]")
_RE_SECTION_HDR  = re.compile(
    r"^(Abstract|Introduction|Conclusion[s]?|Methods?|Results|Discussion|"
    r"Background|Overview|Summary|Related Work|Future Work|"
    r"Impact on Crop Yields|Vulnerable Populations|Adaptation Strategies)"
    r"\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE)

_STANDALONE_LABELS = {
    "abstract", "introduction", "conclusion", "conclusions", "summary",
    "overview", "background", "methodology", "methods", "results",
    "discussion", "related work", "future work", "acknowledgements",
    "acknowledgement",
}
_TRIVIAL = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with", "by", "at"
}

# Verbs that confirm a new sentence is starting inside a merged string
_SPLIT_VERBS = {
    "is", "are", "was", "were", "has", "have", "represents",
    "enables", "provides", "requires", "allows", "shows",
    "developed", "emerged", "created", "built", "found",
}

# Common English verbs used by the title-merge detector
_COMMON_VERBS = {
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "be", "been", "being", "include", "includes", "included", "represent",
    "represents", "allow", "allows", "provide", "provides", "show", "shows",
    "help", "helps", "make", "makes", "give", "gives", "take", "takes",
    "use", "uses", "mean", "means", "find", "found",
}


# ═════════════════════════════════════════════════════════════════════════════
# TITLE-MERGE DETECTOR  (shared with summarizer.py)
# ═════════════════════════════════════════════════════════════════════════════

def _has_title_merge(s: str) -> bool:
    """
    Detect sentences where a document section title has been concatenated
    directly onto the section body with no period separator.

    Examples caught:
      "The Psychology of Human Behaviour and Decision Human behaviour is…"
      "Motivation and the Science of Wellbeing Understanding what motivates…"
      "God through spatial grandeur and artistic richness, and stand as…"
    """
    words = s.split()
    if len(words) < 8:
        return False

    title_count = 0
    for i, w in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", w)
        if not clean:
            continue
        cl = clean.lower()
        if cl in _TRIVIAL:
            continue
        if clean[0].isupper():
            title_count += 1
        elif clean[0].islower():
            break  # lower-case content word ends the title prefix

        if title_count >= 3 and i + 1 < len(words):
            nxt_clean = re.sub(r"[^a-zA-Z]", "", words[i + 1])
            if nxt_clean and nxt_clean[0].isupper() and nxt_clean.lower() not in _TRIVIAL:
                prefix_words = [re.sub(r"[^a-z]", "", w2.lower())
                                for w2 in words[:i + 1]]
                if not any(pw in _COMMON_VERBS for pw in prefix_words):
                    return True

    if len(words) >= 6:
        first_clean = re.sub(r"[^a-zA-Z]", "", words[0])
        if (first_clean and first_clean[0].isupper()
                and len(first_clean) <= 5
                and first_clean.lower() not in _TRIVIAL):
            first_four = [re.sub(r"[^a-z]", "", w.lower()) for w in words[:4]]
            if not any(fw in _COMMON_VERBS for fw in first_four):
                return True

    return False


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def _preprocess(text: str) -> str:
    """Remove standalone section headers before NLTK tokenises."""
    lines  = text.split("\n")
    result = []
    for line in lines:
        s = line.strip()
        if not s:
            result.append(line)
            continue
        if _RE_SECTION_HDR.match(s):
            continue
        words     = s.split()
        ends_sent = s[-1] in ".!?" if s else False
        caps      = sum(1 for w in words if w and w[0].isupper())
        cap_ratio = caps / len(words) if words else 0

        if not ends_sent and cap_ratio >= 0.55 and ":" in s:
            continue

        non_trivial_caps = sum(
            1 for w in words
            if w and w[0].isupper() and w.lower() not in _TRIVIAL)
        if (not ends_sent and len(words) <= 8
                and cap_ratio >= 0.60 and non_trivial_caps >= 2):
            continue

        result.append(line)
    return "\n".join(result)


# ═════════════════════════════════════════════════════════════════════════════
# SENTENCE DISPLAY CLEANING
# ═════════════════════════════════════════════════════════════════════════════

def _clean_for_display(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"[ \t]+",      " ",    s)
    s = re.sub(r",([^\s])",    r", \1", s)
    s = re.sub(r"\.([A-Za-z])", r". \1", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.strip()


# ═════════════════════════════════════════════════════════════════════════════
# SENTENCE QUALITY FILTER
# ═════════════════════════════════════════════════════════════════════════════

def _is_bad(s: str) -> bool:
    """Return True if the sentence should be excluded from key-sentence output."""
    s     = s.strip()
    words = s.split()

    if len(words) < 12:
        return True
    if _RE_ALLCAPS.match(s):
        return True
    if _RE_INLINE_LABEL.match(s):
        return True
    if _RE_ALLCAPS_PFX.match(s):
        return True
    if s.rstrip(":").strip().lower() in _STANDALONE_LABELS:
        return True

    if not s.endswith((".", "!", "?")):
        caps = sum(1 for w in words if w and w[0].isupper())
        if caps / len(words) >= 0.5 and ":" in s:
            return True

    for word in words:
        if len(word) > 20 and "-" not in word and word.isalpha():
            return True

    if re.search(r"\[\d+\]", s):
        return True
    if words and re.match(r"^(19|20)\d{2}$", words[0]):
        return True

    if re.search(r",\s+the\s+[A-Z][a-z]+\.?\s*$", s):
        return True
    if re.search(r",\s+[A-Z][a-z]{3,}\.?\s*$", s) and len(words) < 25:
        return True

    for i in range(3, min(9, len(words))):
        prefix = words[:i]
        if all(w[0].isupper() for w in prefix if w and w.isalpha()):
            lowers = [w for w in prefix
                      if w and w.isalpha() and w[0].islower()
                      and w not in {"and", "or", "the", "a", "an", "of",
                                    "in", "to", "for", "with", "at", "by"}]
            if not lowers and i < len(words) and words[i] and words[i][0].isupper():
                return True

    # Reject title-merged sentences
    if _has_title_merge(s):
        return True

    return False


# ═════════════════════════════════════════════════════════════════════════════
# MERGED-SENTENCE SPLITTER
# ═════════════════════════════════════════════════════════════════════════════

def _split_merged(s: str) -> str:
    """
    Detect two sentences merged without a period and return only the first.
    E.g. "…economic Artificial Intelligence (AI) represents…"
    """
    words = s.split()
    if len(words) < 15:
        return s
    for i in range(8, len(words) - 3):
        w    = words[i]
        prev = words[i - 1] if i > 0 else ""
        if not prev:
            continue
        if (w and w[0].isupper() and len(w) > 3
                and prev[-1] not in ".!?"
                and prev[0].islower()
                and w not in {"The", "A", "An", "In", "On", "At",
                              "By", "For", "Of"}):
            upcoming = [x.lower() for x in words[i: i + 6]]
            if any(v in upcoming for v in _SPLIT_VERBS):
                return " ".join(words[:i]).strip()
    return s


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def get_important_sentences(
    original_text: str,
    summary:       str,
    top_n:         int = 3,
    query:         str = "",
) -> list:
    """
    Return the top N sentences from original_text most relevant to summary.
    Uses zone-based selection for whole-document coverage.

    Args:
        original_text : Cleaned source text (last_input_clean from session state).
        summary       : Generated summary to score against.
        top_n         : Number of key sentences to return (default 3).
        query         : Optional query string; matching words are highlighted.

    Returns:
        List of clean sentence strings in reading order.
    """
    try:
        # Step 1: preprocess — strip standalone headings
        text      = _preprocess(original_text.strip())
        sentences = nltk.sent_tokenize(text)

        # Step 2: clean, split merged sentences, then filter
        cleaned_sents: list = []
        for s in sentences:
            s = _clean_for_display(s.strip())
            s = _split_merged(s)
            if not _is_bad(s):
                cleaned_sents.append(s)
        sentences = cleaned_sents

        if not sentences:
            return ["No key sentences found in the source text."]
        if len(sentences) <= top_n:
            return sentences

        # Step 3: TF-IDF relevance against the summary
        all_texts  = [summary] + sentences
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf      = vectorizer.fit_transform(all_texts)
        rel_scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

        n = len(sentences)

        # Step 4: zone-based selection (intro / middle / end)
        zone_sz  = max(1, n // top_n)
        selected: list = []

        for z in range(top_n):
            start = z * zone_sz
            end   = min(n, start + zone_sz) if z < top_n - 1 else n
            if start >= n:
                break
            zone = list(range(start, end))
            best = max(zone, key=lambda i: rel_scores[i])
            selected.append(best)

        # Step 5: deduplicate selected sentences
        _STOP = {
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
            "and", "or", "that", "this", "it", "for", "with", "has", "have",
            "been",
        }
        kept:   list = []
        kept_w: list = []
        for idx in selected:
            sw = set(sentences[idx].lower().split()) - _STOP
            # max(…,1) prevents ZeroDivisionError on empty content sets
            if not any(len(sw & ew) / max(len(sw), len(ew), 1) >= 0.60
                       for ew in kept_w):
                kept.append(idx)
                kept_w.append(sw)

        # Step 6: return in reading order
        result = [sentences[i] for i in sorted(kept)]

        # Optional: highlight query keywords
        if query and query.strip():
            _STOP_Q = {
                "what", "how", "why", "when", "where", "which", "who",
                "the", "a", "an", "is", "are", "was", "were", "of", "in",
                "to", "and", "or", "for", "with", "about", "does", "do",
                "did", "has", "have", "had", "can", "will", "should",
            }
            q_words = [
                w.lower().strip("?.,!")
                for w in query.split()
                if w.lower().strip("?.,!") not in _STOP_Q and len(w) > 2
            ]
            highlighted: list = []
            for sent in result:
                hl = sent
                for word in q_words:
                    pat = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                    hl  = pat.sub(
                        r'<mark style="background:rgba(255,210,0,0.45);'
                        r'border-radius:3px;padding:0 2px;font-weight:600;">'
                        r"\1</mark>",
                        hl,
                    )
                highlighted.append(hl)
            return highlighted

        return result

    except Exception as exc:
        # Always return a list so the caller can safely iterate
        return [f"Could not extract key sentences: {exc}"]