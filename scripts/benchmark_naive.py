#!/usr/bin/env python3
"""
Naive baseline benchmark for LLM inference (no batching).
Processes prompts sequentially, one at a time.
"""

import json
import time
import csv
from datetime import datetime
from typing import List, Dict, Tuple
import random

import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not found. Install with: pip install transformers")
    exit(1)


# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 128
PRECISION = torch.float16
SEEDS = [42, 43, 44]
WORKLOAD_FILE = "workload.json"
RESULTS_CSV = "results_naive.csv"
RESULTS_META = "results_naive_meta.json"


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_gpu_cache():
    """Clear GPU cache and reset memory stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_gpu_memory_gb() -> float:
    """Get peak GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def load_workload(filepath: str) -> List[Dict]:
    """Load workload from JSON file."""
    print(f"Loading workload from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)

    prompts = data["prompts"]
    print(f"  Loaded {len(prompts)} prompts")
    print(f"  Short prompts: {sum(1 for p in prompts if p['category'] == 'short')}")
    print(f"  Long prompts: {sum(1 for p in prompts if p['category'] == 'long')}")
    return prompts


def load_model(model_name: str, precision: torch.dtype):
    """
    Load model and tokenizer.

    Args:
        model_name: HuggingFace model name
        precision: torch dtype (e.g., torch.float16)

    Returns:
        (model, tokenizer)
    """
    print(f"\nLoading model: {model_name}")
    print(f"  Precision: {precision}")

    try:
        # Load tokenizer
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=precision,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        model.eval()  # Set to evaluation mode

        print(f"  ✓ Model loaded successfully")

        # Print device info
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  Device: {device_name}")
        else:
            print("  Device: CPU (WARNING: This will be very slow!)")

        return model, tokenizer

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nPossible solutions:")
        print("  1. Check your internet connection")
        print("  2. Verify you have enough GPU memory")
        print("  3. Try: huggingface-cli login")
        print("  4. Check model name is correct")
        raise


def run_inference(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int
) -> Tuple[float, int]:
    """
    Run inference for a single prompt.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt_text: Input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        (latency_ms, tokens_generated)
    """
    # Tokenize input
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    # Measure inference time
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    end_time = time.perf_counter()

    # Calculate metrics
    latency_s = end_time - start_time
    latency_ms = latency_s * 1000

    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length

    return latency_ms, tokens_generated


def run_benchmark_for_seed(
    seed: int,
    model,
    tokenizer,
    prompts: List[Dict],
    max_new_tokens: int
) -> Dict:
    """
    Run benchmark for a single seed.

    Args:
        seed: Random seed
        model: The loaded model
        tokenizer: The tokenizer
        prompts: List of prompt dictionaries
        max_new_tokens: Max tokens to generate per prompt

    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print(f"Running benchmark with seed {seed}")
    print(f"{'='*60}")

    # Set seeds and clear GPU
    set_seeds(seed)
    clear_gpu_cache()

    # Results storage
    results = []

    # Process each prompt
    total_start = time.perf_counter()

    with tqdm(total=len(prompts), desc=f"Seed {seed}", unit="prompt") as pbar:
        for prompt_data in prompts:
            prompt_id = prompt_data["id"]
            prompt_text = prompt_data["prompt"]

            # Run inference
            latency_ms, tokens_generated = run_inference(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens
            )

            # Store result
            results.append({
                "prompt_id": prompt_id,
                "latency_ms": latency_ms,
                "tokens_generated": tokens_generated
            })

            pbar.update(1)

    total_end = time.perf_counter()
    total_time_s = total_end - total_start

    # Calculate metrics
    latencies = [r["latency_ms"] for r in results]
    tokens_list = [r["tokens_generated"] for r in results]

    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    total_tokens = sum(tokens_list)
    throughput_tok_s = total_tokens / total_time_s

    peak_memory_gb = get_gpu_memory_gb()

    metrics = {
        "seed": seed,
        "p50_ms": p50_latency,
        "p95_ms": p95_latency,
        "p99_ms": p99_latency,
        "throughput_tok_s": throughput_tok_s,
        "peak_mem_gb": peak_memory_gb,
        "total_time_s": total_time_s,
        "total_tokens": total_tokens,
        "num_prompts": len(prompts)
    }

    # Print summary
    print(f"\nSeed {seed} Results:")
    print("-" * 40)
    print(f"p50 latency:    {p50_latency:.1f} ms")
    print(f"p95 latency:    {p95_latency:.1f} ms")
    print(f"p99 latency:    {p99_latency:.1f} ms")
    print(f"Throughput:     {throughput_tok_s:.1f} tok/s")
    print(f"Peak Memory:    {peak_memory_gb:.1f} GB")
    print(f"Total Time:     {total_time_s:.1f} s")
    print("-" * 40)

    return metrics


def save_results_csv(results: List[Dict], filepath: str):
    """Save results to CSV file."""
    print(f"\nSaving results to {filepath}...")

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'seed',
            'p50_ms',
            'p95_ms',
            'p99_ms',
            'throughput_tok_s',
            'peak_mem_gb',
            'total_time_s'
        ])

        # Write data
        for result in results:
            writer.writerow([
                result['seed'],
                f"{result['p50_ms']:.1f}",
                f"{result['p95_ms']:.1f}",
                f"{result['p99_ms']:.1f}",
                f"{result['throughput_tok_s']:.1f}",
                f"{result['peak_mem_gb']:.1f}",
                f"{result['total_time_s']:.1f}"
            ])

    print(f"  ✓ Results saved")


def save_metadata(prompts: List[Dict], filepath: str):
    """Save benchmark metadata to JSON file."""
    print(f"Saving metadata to {filepath}...")

    metadata = {
        "model_name": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "precision": str(PRECISION),
        "num_prompts": len(prompts),
        "seeds": SEEDS,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["cuda_version"] = torch.version.cuda
        metadata["num_gpus"] = torch.cuda.device_count()
    else:
        metadata["gpu_name"] = "CPU"
        metadata["cuda_version"] = None
        metadata["num_gpus"] = 0

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata saved")


def main():
    """Main execution function."""
    print("="*60)
    print("NAIVE BASELINE BENCHMARK")
    print("="*60)

    # Load workload
    prompts = load_workload(WORKLOAD_FILE)

    # Load model
    model, tokenizer = load_model(MODEL_NAME, PRECISION)

    # Save metadata
    save_metadata(prompts, RESULTS_META)

    # Run benchmarks for each seed
    all_results = []

    for seed in SEEDS:
        try:
            metrics = run_benchmark_for_seed(
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=MAX_NEW_TOKENS
            )

            all_results.append(metrics)

            # Save CSV after each seed (in case of crash)
            save_results_csv(all_results, RESULTS_CSV)

        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user.")
            if all_results:
                save_results_csv(all_results, RESULTS_CSV)
            print("Partial results saved.")
            exit(1)

        except Exception as e:
            print(f"\n✗ Error during benchmark with seed {seed}: {e}")
            if all_results:
                save_results_csv(all_results, RESULTS_CSV)
            raise

    # Final summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

    print(f"\nProcessed {len(prompts)} prompts across {len(SEEDS)} seeds")
    print(f"Results saved to: {RESULTS_CSV}")
    print(f"Metadata saved to: {RESULTS_META}")

    # Aggregate statistics
    if len(all_results) > 1:
        print("\nAggregate Statistics (across all seeds):")
        print("-" * 40)

        avg_p50 = np.mean([r['p50_ms'] for r in all_results])
        avg_p95 = np.mean([r['p95_ms'] for r in all_results])
        avg_p99 = np.mean([r['p99_ms'] for r in all_results])
        avg_throughput = np.mean([r['throughput_tok_s'] for r in all_results])
        avg_memory = np.mean([r['peak_mem_gb'] for r in all_results])

        print(f"Avg p50 latency:    {avg_p50:.1f} ms (±{np.std([r['p50_ms'] for r in all_results]):.1f})")
        print(f"Avg p95 latency:    {avg_p95:.1f} ms (±{np.std([r['p95_ms'] for r in all_results]):.1f})")
        print(f"Avg p99 latency:    {avg_p99:.1f} ms (±{np.std([r['p99_ms'] for r in all_results]):.1f})")
        print(f"Avg throughput:     {avg_throughput:.1f} tok/s (±{np.std([r['throughput_tok_s'] for r in all_results]):.1f})")
        print(f"Avg peak memory:    {avg_memory:.1f} GB (±{np.std([r['peak_mem_gb'] for r in all_results]):.1f})")
        print("-" * 40)

    print("\n✓ Benchmark completed successfully!")


if __name__ == "__main__":
    main()
