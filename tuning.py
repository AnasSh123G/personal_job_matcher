import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.1"

os.environ["HSA_ENABLE_SDMA"] = "0"

os.environ["HIP_VISIBLE_DEVICES"] = "0"

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch

# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.95, device=0)

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train_job_matcher(
    model_id,
    output_dir,
    final_model_name,
    num_train_epochs,
    r,
    lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    quantize=False, 
    resume_from_checkpoint=None
):
    with open('output.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted_data = []

    RESUME_SUMMARY = """
CV summary here
"""

    SYSTEM_PROMPT = f"""You are an AI job matcher evaluating job descriptions against a specific resume. 
The candidate's resume summary is as follows:\n\n
{RESUME_SUMMARY}

Your task is to analyze the provided job description and determine if this job is recommended for the candidate.
Please output ONLY the following format: "Recommended: yes" or "Recommended: no"."""

    for item in raw_data:
        user_msg = f"Job Description:\n{item['job_description']}"
        
        answer = "yes" if item["recommended"] else "no"
        expected_output = f"Recommended: {answer}"
        
        formatted_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": expected_output}
            ]
        })

    dataset = Dataset.from_list(formatted_data)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, enable_thinking=False)
        tokenized = tokenizer(text, truncation=True, max_length=3500)
        
        prompt_messages = example["messages"][:-1]
        
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_text += "Recommended:"
        prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=3500)
        
        prompt_len = len(prompt_tokenized["input_ids"])
        
        labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
        labels = labels[:len(tokenized["input_ids"])] 
        
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(preprocess_function, batched=False, remove_columns=["messages"])

    max_seq_len = 3500
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_len)

    dataset = dataset.train_test_split(test_size=0.1)

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )

    peft_config = LoraConfig(
        r=r, 
        lora_alpha=lora_alpha, 
        target_modules=target_modules, 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=16, 
        num_train_epochs=num_train_epochs, 
        learning_rate=5e-5, 
        lr_scheduler_type="cosine", 
        warmup_ratio=0.1, 
        weight_decay=0.01, 
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", 
        greater_is_better=False, 
        fp16=True, 
        bf16=False,
        optim="paged_adamw_8bit", 
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        remove_unused_columns=False, 
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=collator,
    )
    trainer.model.print_trainable_parameters()

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("Starting training! Keep an eye on your VRAM usage...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.model.save_pretrained(final_model_name)
    tokenizer.save_pretrained(final_model_name)
    print("Training complete! Adapter saved.")
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model for job matching.")
    parser.add_argument("--model_id", type=str, default="1B_lora_job_matcher/checkpoint-600", help="Base model ID or path")
    parser.add_argument("--output_dir", type=str, default="./1B_lora_job_matcher", help="Output directory for checkpoints")
    parser.add_argument("--final_model_name", type=str, default="1B_final_job_matcher", help="Name for the final saved model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in output_dir")
    args = parser.parse_args()

    # train_job_matcher(
    #     # model_id="Qwen/Qwen3.5-4B",
    #     model_id="CohereLabs/tiny-aya-water",
    #     output_dir="./tiny-aya-water_lora_job_matcher",
    #     final_model_name="tiny-aya-water_final_job_matcher",
    #     num_train_epochs=8,
    #     r=32,
    #     lora_alpha=64,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     quantize=False, # 4-bit quantization required for 9B model on 16GB VRAM
    #     resume_from_checkpoint="./tiny-aya-water_lora_job_matcher/checkpoint-215" # Fresh start with new model and binary scoring
    # )
