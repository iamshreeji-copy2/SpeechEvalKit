from __future__ import annotations


def mer_score(reference: str, hypothesis: str) -> float:
    """
    Compute Match Error Rate.

    Requires optional dependency:
        pip install jiwer
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError(
            "MER requires optional dependency 'jiwer'. "
            "Install with: pip install jiwer"
        ) from exc

    return float(jiwer.mer(reference, hypothesis))
