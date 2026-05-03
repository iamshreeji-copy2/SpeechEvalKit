from speechevalkit.gui import launch_gui

launch_gui(
    audio_dir="generated/",
    ref_dir="ground_truth/",
    output_csv="smos_scores.csv",
    mode="smos",
)
