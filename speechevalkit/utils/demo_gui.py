from speechevalkit.gui import launch_gui

launch_gui(
    audio_dir="generated/",
    output_csv="mos_scores.csv",
    mode="mos",
)
