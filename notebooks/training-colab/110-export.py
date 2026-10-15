# @title 10. Export to GGUF (best-effort)
if not RUN_EXPORT:
    print("RUN_EXPORT is disabled; skipping merge and GGUF conversion.")
else:
    ok = run_step("merge", f"python app.py export merge --config {CONFIG} "
                           f"--adapter {ADAPTER_DIR} --out {MERGED_DIR}", allow_fail=True) == 0
    if ok:
        # A CPU build is enough for convert + quantize and compiles far faster than CUDA.
        ok = run_step("build-llama-cpp", "bash scripts/build_llama_cpp.sh",
                      env={"BUILD_CUDA": "0"}, allow_fail=True) == 0
    if ok:
        ok = run_step("gguf", f"python app.py export gguf --merged {MERGED_DIR} --quant Q4_K_M",
                      allow_fail=True) == 0
    if ok:
        for gguf in sorted(Path("artifacts/gguf").glob(f"{RUN_NAME}-*.gguf*")):
            if gguf.suffix == ".json":
                archive_results([gguf])
            elif UPLOAD_GGUF:
                print(f"Uploading {gguf.name} ({gguf.stat().st_size / 1e9:.2f} GB) to the Hub...")
                hub_upload(gguf, f"gguf/{RUN_NAME}/{gguf.name}", blocking=True)
    else:
        print("Export did not complete; see the logs above. Training and evaluation results are unaffected.")
