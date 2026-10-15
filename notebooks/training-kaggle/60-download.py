# 5. Download the dataset and the base model(s), one at a time
for target in DOWNLOAD_ORDER:
    run_step(f"download-{target}", f"bash download_data.sh {target}")
