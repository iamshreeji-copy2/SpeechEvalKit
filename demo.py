from speechevalkit import evaluate

results = evaluate(
    ref_dir="/home/aaa/Documents/Drashti_Savaj/Torgo/Torgo/F/F01/Session1/wav_headMic/",
    pred_dir="/home/aaa/Documents/Drashti_Savaj/Torgo/Torgo/F/F01/Session1/wav_headMic/",
    metrics=["pesq", "stoi", "mcd", "si_sdr", "cosine"],
)

print(results["summary"])
