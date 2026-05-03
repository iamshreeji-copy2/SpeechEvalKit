from __future__ import annotations


def cer_score(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate.

    Requires optional dependency:
        pip install jiwer
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError(
            "CER requires optional dependency 'jiwer'. "
            "Install with: pip install jiwer"
        ) from exc

    return float(jiwer.cer(reference, hypothesis))
