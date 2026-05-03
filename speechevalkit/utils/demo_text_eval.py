from speechevalkit.metrics.wer import wer_score
from speechevalkit.metrics.cer import cer_score
from speechevalkit.metrics.mer import mer_score

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jump over lazy dog"

print("WER:", wer_score(reference, hypothesis))
print("CER:", cer_score(reference, hypothesis))
print("MER:", mer_score(reference, hypothesis))
