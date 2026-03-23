#!/usr/bin/env python3
import argparse
from create_dataset import create_listings_dataset, levels, fields, header, industries, work_connditions, salaries, vibes
from convert_to_json import convert_csv_to_json, clean_job_data, scoring
from tuning import train_job_matcher
from benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="CLI for Job Matcher project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create_dataset command
    parser_create = subparsers.add_parser("create_dataset", help="Create synthetic job listings dataset")
    parser_create.add_argument("--output", type=str, default="listings_anthropic.csv", help="Output CSV file name")
    parser_create.add_argument("--num_rows", type=int, default=400, help="Number of rows to generate")
    parser_create.add_argument("--api_provider", type=str, choices=["perplexity", "anthropic"], default="anthropic", help="API Provider")
    parser_create.add_argument("--model", type=str, default="claude-haiku-4-5-20251001", help="Model name")

    # process_data command
    parser_process = subparsers.add_parser("process", help="Convert CSV to JSON, clean, and score")
    parser_process.add_argument("--input_csv", type=str, default="listings_anthropic.csv", help="Input CSV file")
    parser_process.add_argument("--output_json", type=str, default="output_anthropic.json", help="Output JSON file")
    parser_process.add_argument("--no_convert", action="store_true", help="Skip CSV to JSON conversion")
    parser_process.add_argument("--no_clean", action="store_true", help="Skip cleaning")
    parser_process.add_argument("--no_score", action="store_true", help="Skip scoring")

    # tune command
    parser_tune = subparsers.add_parser("tune", help="Fine-tune the model")
    parser_tune.add_argument("--model_id", type=str, default="CohereLabs/tiny-aya-water", help="Base model ID or path")
    parser_tune.add_argument("--output_dir", type=str, default="./tiny-aya-water_lora_job_matcher", help="Output directory for checkpoints")
    parser_tune.add_argument("--final_model_name", type=str, default="tiny-aya-water_final_job_matcher", help="Name for the final saved model")
    parser_tune.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser_tune.add_argument("--resume", type=str, default=None, help="Resume from a specific checkpoint path")
    parser_tune.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization")

    # benchmark command
    parser_bench = subparsers.add_parser("benchmark", help="Run benchmark on model")
    parser_bench.add_argument("--data", type=str, default="output_testing.json", help="Test JSON data")
    parser_bench.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-2B", help="Base model path")
    parser_bench.add_argument("--adapter", type=str, default="./Qwen3.5-2B_final_job_matcher", help="Adapter path")
    parser_bench.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser_bench.add_argument("--quantize", action="store_true", help="Enable quantization")

    args = parser.parse_args()

    if args.command == "create_dataset":
        print(f"Creating dataset {args.output} with {args.num_rows} rows using {args.api_provider}...")
        create_listings_dataset(args.output, args.num_rows, levels, fields, header, industries, work_connditions, salaries, vibes, api_provider=args.api_provider, model=args.model)
        print("Dataset created.")

    elif args.command == "process":
        if not args.no_convert:
            print(f"Converting {args.input_csv} to {args.output_json}...")
            convert_csv_to_json(args.input_csv, args.output_json)
        if not args.no_clean:
            print(f"Cleaning {args.output_json}...")
            clean_job_data(args.output_json)
        if not args.no_score:
            print(f"Scoring {args.output_json}...")
            scoring(args.output_json)
        print("Data processing complete.")

    elif args.command == "tune":
        print(f"Starting tuning for {args.model_id}...")
        train_job_matcher(
            model_id=args.model_id,
            output_dir=args.output_dir,
            final_model_name=args.final_model_name,
            num_train_epochs=args.epochs,
            resume_from_checkpoint=args.resume,
            quantize=args.quantize
        )

    elif args.command == "benchmark":
        print(f"Running benchmark on {args.model_path} with adapter {args.adapter}...")
        run_benchmark(
            data=args.data,
            model_path=args.model_path,
            adapter=args.adapter,
            limit=args.limit,
            quantize=args.quantize
        )

if __name__ == "__main__":
    main()
