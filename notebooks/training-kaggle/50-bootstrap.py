# 4. Install dependencies (colab_bootstrap.sh is platform-neutral: pip + hf CLI check)
run_step("bootstrap", "bash scripts/colab_bootstrap.sh",
         env={"INSTALL_UNSLOTH": "1" if INSTALL_UNSLOTH else "0"})
