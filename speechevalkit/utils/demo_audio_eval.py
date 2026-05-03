from speechevalkit import evaluate

results = evaluate(
    ref_dir="ground_truth/",
    pred_dir="generated/",
    metrics=["pesq", "stoi", "mcd", "si_sdr", "cosine"],
    sample_rate=16000,
    show_progress=True,
    save_json="audio_results.json",
)

print(results["summary"])
