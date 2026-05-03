from __future__ import annotations


def wer_score(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate.

    Requires optional dependency:
        pip install jiwer
    """
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError(
            "WER requires optional dependency 'jiwer'. "
            "Install with: pip install jiwer"
        ) from exc

    return float(jiwer.wer(reference, hypothesis))
