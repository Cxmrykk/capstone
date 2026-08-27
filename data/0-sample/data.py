import json
import os
import warnings
from datasets import load_dataset

# Suppress Hugging Face warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

# Dynamically find paths based on the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, ".."))

print(f"Working Directory: {script_dir}")
print(f"Data Directory: {data_dir}\n")

# --- 1. Save Dataset Sample ---
parquet_path = os.path.join(data_dir, "text2cypher-2024v1", "data", "train-00000-of-00001.parquet")
if os.path.exists(parquet_path):
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    # Save the first row
    with open(os.path.join(script_dir, "dataset_sample.json"), "w", encoding="utf-8") as f:
        json.dump(dataset[0], f, indent=4)
    print("✅ Saved: dataset_sample.json")
else:
    print(f"❌ Missing: {parquet_path}")

# --- Helper Function for Config Files ---
def save_config_as_json(rel_in_path, out_filename, is_json=True):
    in_path = os.path.join(data_dir, rel_in_path)
    out_path = os.path.join(script_dir, out_filename)
    
    if os.path.exists(in_path):
        with open(in_path, "r", encoding="utf-8") as f:
            if is_json:
                content = json.load(f)
            else:
                # Wrap raw text (like .jinja) inside a JSON object
                content = {"content": f.read()}
                
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=4)
        print(f"✅ Saved: {out_filename}")
    else:
        print(f"❌ Missing: {rel_in_path}")

# --- 2. Save Model Configs ---
save_config_as_json("Qwen3.5-4B/config.json", "qwen_config.json")
save_config_as_json("Qwen3.5-4B/chat_template.jinja", "qwen_chat_template.json", is_json=False)
save_config_as_json("gemma-4-E4B/config.json", "gemma_config.json")
save_config_as_json("gemma-4-E4B/tokenizer_config.json", "gemma_tokenizer_config.json")

print("\nDone! Check the data/0-sample/ folder for the output files.")