import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.1"
os.environ["HSA_ENABLE_SDMA"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch

# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.95, device=0)

import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

RESUME_SUMMARY = """
CV Summary here
"""

SYSTEM_PROMPT = f"""You are an AI job matcher evaluating job descriptions against a specific resume. 
The candidate's resume summary is as follows:\n\n
{RESUME_SUMMARY}

Your task is to analyze the provided job description and determine if this job is recommended for the candidate.
Please output ONLY the following format: "Recommended: yes" or "Recommended: no"."""

class BenchmarkArgs:
    def __init__(self, data, model_path, adapter, limit, quantize):
        self.data = data
        self.model_path = model_path
        self.adapter = adapter
        self.limit = limit
        self.quantize = quantize

def run_benchmark(data, model_path, adapter=None, limit=None, quantize=False):
    args = BenchmarkArgs(data, model_path, adapter, limit, quantize)
    
    print(f"Loading tokenizer {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model -> {args.model_path} (Quantize: {args.quantize})...")
    
    if args.quantize:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    if args.adapter and args.adapter.strip():
        print(f"Loading LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(base_model, args.adapter)
    else:
        print("No adapter specified. Running base model only.")
        model = base_model
        
    model.eval()

    print(f"Loading data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]

    correct_predictions = 0
    total_processed = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    print(f"\nStarting Benchmark for Binary Recommendation...\n")
    
    for i, item in enumerate(data):
        job_desc = item.get("job_description", "")
        actual_recommended = item.get("recommended", None)
        
        if actual_recommended is None:
            continue
            
        user_msg = f"Job Description:\n{job_desc}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        prompt += "Recommended:"
        
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        if '</think>' in generated_text:
            generated_text = generated_text.split('</think>')[-1].strip()
        
        predicted_recommended = None
        lower_text = generated_text.lower()
        if "yes" in lower_text:
            predicted_recommended = True
        elif "no" in lower_text:
            predicted_recommended = False

        actual_str = "yes" if actual_recommended else "no"
        if predicted_recommended is not None:
            pred_str = "yes" if predicted_recommended else "no"
            is_correct = (predicted_recommended == actual_recommended)
            
            if is_correct:
                correct_predictions += 1
                
            if predicted_recommended:
                if actual_recommended:
                    tp += 1
                else:
                    fp += 1
            else:
                if actual_recommended:
                    fn += 1
                else:
                    tn += 1
                
            status = "PASS" if is_correct else "FAIL"
            print(f"[{i+1}/{len(data)}] Actual: {actual_str} | Predicted: {pred_str} | {status}")
            if not is_correct:
                print(f"    Raw Output: {repr(generated_text)}")
        else:
            print(f"[{i+1}/{len(data)}] Actual: {actual_str} | Predicted: None | FAIL \n    Raw Output: {repr(generated_text)}")
            
        total_processed += 1

    if total_processed > 0:
        accuracy = (correct_predictions / total_processed) * 100
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n============================")
        print(f"     BENCHMARK RESULTS      ")
        print(f"============================")
        print(f"Total Evaluated : {total_processed}")
        print(f"Correct         : {correct_predictions}")
        print(f"Accuracy        : {accuracy:.2f}%")
        print(f"----------------------------")
        print(f"True Positives  : {tp}")
        print(f"True Negatives  : {tn}")
        print(f"False Positives : {fp}")
        print(f"False Negatives : {fn}")
        print(f"----------------------------")
        print(f"Precision       : {precision:.4f}")
        print(f"Recall          : {recall:.4f}")
        print(f"F1 Score        : {f1_score:.4f}")
        print(f"============================")

        try:
            import matplotlib.pyplot as plt
            import os
            model_name = model_path.replace("/", "_")
            adapter_suffix = "_with_adapter" if (args.adapter and args.adapter.strip()) else "_base"
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [accuracy / 100, precision, recall, f1_score]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
            
            plt.ylim(0, 1.05)
            plt.ylabel('Score')
            plt.title(f'Benchmark Performance Metrics for {model_name}{adapter_suffix}')
            
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
                
            
            plot_path = os.path.join(os.getcwd(), f'{model_name}{adapter_suffix}_benchmark_metrics.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"\nMetrics plot saved successfully to: {plot_path}")
        except ImportError:
            print("\nNote: 'matplotlib' is not installed. Skipping plot generation. Run 'pip install matplotlib' to enable it.")

    else:
        print("No valid data processed.")

# if __name__ == "__main__":

#     run_benchmark(
#         data="benchmark_set.json", 
#         model_path="Qwen/Qwen3.5-2B", 
#         adapter="./Qwen3.5-2B_final_job_matcher", 
#         limit=None,
#         quantize=False
#     )
