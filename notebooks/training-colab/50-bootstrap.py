# @title 4. Install dependencies
run_step("bootstrap", "bash scripts/colab_bootstrap.sh",
         env={"INSTALL_UNSLOTH": "1" if INSTALL_UNSLOTH else "0"})
