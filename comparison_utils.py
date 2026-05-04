from difflib import SequenceMatcher

from text_utils import clean_text


def tokenize_text(text):
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def build_bigrams(tokens):
    if len(tokens) < 2:
        return set()
    return {" ".join(tokens[index : index + 2]) for index in range(len(tokens) - 1)}


def longest_token_streak(claim_tokens, evidence_tokens):
    evidence_set = set(evidence_tokens)
    best_streak = 0
    current_streak = 0

    for token in claim_tokens:
        if token in evidence_set:
            current_streak += 1
            best_streak = max(best_streak, current_streak)
        else:
            current_streak = 0

    return best_streak


def compute_similarity_metrics(claim_text, evidence_text):
    claim_tokens = tokenize_text(claim_text)
    evidence_tokens = tokenize_text(evidence_text)
    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    overlap = claim_set.intersection(evidence_set)
    union = claim_set.union(evidence_set)
    claim_bigrams = build_bigrams(claim_tokens)
    evidence_bigrams = build_bigrams(evidence_tokens)
    bigram_overlap = claim_bigrams.intersection(evidence_bigrams)
    claim_word_count = len(claim_tokens)
    evidence_word_count = len(evidence_tokens)
    first_terms = claim_tokens[: min(12, claim_word_count)]
    leading_hits = sum(1 for token in first_terms if token in evidence_set)
    cleaned_claim = " ".join(claim_tokens)
    cleaned_evidence = " ".join(evidence_tokens)

    overlap_count = len(overlap)
    claim_coverage = overlap_count / claim_word_count if claim_word_count else 0.0
    evidence_precision = overlap_count / evidence_word_count if evidence_word_count else 0.0
    jaccard_score = overlap_count / len(union) if union else 0.0
    bigram_score = (
        len(bigram_overlap) / len(claim_bigrams)
        if claim_bigrams
        else 0.0
    )
    leading_term_score = leading_hits / len(first_terms) if first_terms else 0.0
    streak_score = (
        longest_token_streak(claim_tokens, evidence_tokens) / claim_word_count
        if claim_word_count
        else 0.0
    )
    length_ratio = (
        min(claim_word_count, evidence_word_count) / max(claim_word_count, evidence_word_count)
        if claim_word_count and evidence_word_count
        else 0.0
    )
    sequence_score = (
        SequenceMatcher(None, cleaned_claim, cleaned_evidence).ratio()
        if cleaned_claim and cleaned_evidence
        else 0.0
    )

    return {
        "claim_coverage": round(claim_coverage, 6),
        "evidence_precision": round(evidence_precision, 6),
        "jaccard_score": round(jaccard_score, 6),
        "bigram_score": round(bigram_score, 6),
        "leading_term_score": round(leading_term_score, 6),
        "streak_score": round(streak_score, 6),
        "length_ratio": round(length_ratio, 6),
        "sequence_score": round(sequence_score, 6),
        "claim_word_count": claim_word_count,
        "evidence_word_count": evidence_word_count,
        "overlap_count": overlap_count,
    }


def extract_comparison_features(claim_text, evidence_text):
    metrics = compute_similarity_metrics(claim_text, evidence_text)
    return [
        metrics["claim_coverage"],
        metrics["evidence_precision"],
        metrics["jaccard_score"],
        metrics["bigram_score"],
        metrics["leading_term_score"],
        metrics["streak_score"],
        metrics["length_ratio"],
        metrics["sequence_score"],
        metrics["claim_word_count"],
        metrics["evidence_word_count"],
    ]
