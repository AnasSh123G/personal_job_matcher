# Personalized AI Job Matcher

This project is a personalized AI-driven job matching pipeline tailored specifically for my own profile and skill set. It evaluates job descriptions against my personal resume summary to determine if a role is a good fit. 

The system automatically creates synthetic datasets of job listings, processes the descriptions to determine a match score, fine-tunes a Large Language Model (LLM) using LoRA to predict recommendations locally, and benchmarks the model's predictive accuracy.

**Hardware & Performance:**
This entire pipeline, including the model fine-tuning and inference, was successfully executed locally on **AMD hardware (ROCm)**. 
By fine-tuning the model (like Qwen 3.5 2B) on this personalized dataset, the model's matching accuracy saw a significant improvement—jumping from a baseline of ~65% to **approximately 84%**.


<img width="49%" alt="Qwen_Qwen3 5-2B_base_benchmark_metrics" src="https://github.com/user-attachments/assets/580927e1-5265-43db-a731-56d64610e2a9" /> <img width="49%" alt="Qwen_Qwen3 5-2B_with_adapter_benchmark_metrics" src="https://github.com/user-attachments/assets/ebabdb20-0ea6-4365-ab35-85e011977505" />



## Architecture & Project Structure

- **`main.py`**: A unified command-line interface (CLI) that orchestrates the pipeline.
- **`create_dataset.py`**: Calls Perplexity or Anthropic APIs to generate synthetic job descriptions given various profiles, levels, and fields.
- **`convert_to_json.py`**: Parses the plain text CSVs into JSON format, extracts target skills, calculates missing skills compared to my personal knowledge base, and computes a final match score resulting in a `recommended` (yes/no) label.
- **`tuning.py`**: Fine-tunes language models (like Qwen, Llama, or Aya) on the labeled dataset utilizing QLoRA (4-bit quantization, LoRA adapter![Uploading Qwen_Qwen3.5-2B_base_benchmark_metrics.png…]()
s) with specific optimizations for **AMD ROCm environments** (e.g., handling VRAM limits and allocator configurations).
- **`benchmark.py`**: Runs a deterministic generation benchmark of the tuned model on unseen job listings to calculate Accuracy, Precision, Recall, and F1 Score for its binary classification task.

## Usage

You can iterate through the entire workflow sequentially via `main.py`. Use `./main.py <command> --help` for full parameter details.

### 1. Create a Synthetic Dataset
Generates synthetic job listings in German and outputs them to a CSV file.
```bash
./main.py create_dataset --output listings.csv --num_rows 400 --api_provider anthropic
```

### 2. Process Data
Converts the generated CSV to JSON, cleans the job description text (removing arbitrary prefixes and markdown), and computes the binary recommendation target values.
```bash
./main.py process --input_csv listings.csv --output_json dataset.json
```
*(Options to cherry-pick processing stages: `--no_convert`, `--no_clean`, `--no_score`)*

### 3. Fine-Tune the Model
Fine-tunes the specified base LLM on the processed dataset leveraging Hugging Face Transformers and PEFT frameworks.
```bash
./main.py tune --model_id Qwen/Qwen3.5-2B --output_dir ./lora_output --final_model_name final_job_model --epochs 8
```

### 4. Benchmark the Model
Evaluates the candidate-matching performance of the adapter natively on JSON data.
```bash
./main.py benchmark --data dataset.json --model_path CohereLabs/tiny-aya-water --adapter ./lora_output
```
