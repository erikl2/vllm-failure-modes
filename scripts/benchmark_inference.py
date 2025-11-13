#!/usr/bin/env python3
"""
LLM Inference Benchmarking Script
Runs inference benchmarks with varying prompt lengths
"""

import time
import sys
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict


def check_gpu_availability():
    """Check if GPU is available and print device info"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {gpu_name}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("⚠ No GPU available, falling back to CPU (this will be slow)")

    return device


def generate_prompt(target_length: int, tokenizer) -> str:
    """Generate a prompt with approximately target_length tokens"""
    base_text = "The quick brown fox jumps over the lazy dog. "

    # Estimate words needed (rough approximation: 1 token ≈ 0.75 words)
    words_needed = int(target_length * 0.75)

    # Build prompt by repeating base text
    prompt = ""
    while len(prompt.split()) < words_needed:
        prompt += base_text

    # Verify actual token count
    actual_tokens = len(tokenizer.encode(prompt))

    return prompt, actual_tokens


def run_inference_with_timing(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 50) -> Dict:
    """Run inference and measure timing metrics"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    # Track timing
    start_time = time.time()
    first_token_time = None

    # Run inference
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    total_time = time.time() - start_time

    # Calculate metrics
    output_length = outputs.shape[1]
    tokens_generated = output_length - input_length
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0

    # Time to first token approximation (total_time / tokens_generated for first token)
    # Note: This is approximate without streaming
    time_to_first_token = total_time / tokens_generated if tokens_generated > 0 else total_time

    return {
        "prompt_length": input_length,
        "tokens_generated": tokens_generated,
        "total_time_s": total_time,
        "time_to_first_token_ms": time_to_first_token * 1000,
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": total_time * 1000,
    }


def main():
    # Model configuration
    model_name = "facebook/opt-350m"
    target_prompt_lengths = [50, 100, 200, 500, 1000]
    output_file = "results.csv"

    print("=" * 80)
    print("LLM Inference Benchmark - Extended")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Test prompt lengths: {target_prompt_lengths}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    # Check GPU availability
    device = check_gpu_availability()
    print()

    try:
        # Load tokenizer
        print("Loading tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_load_time = time.time() - start_time
        print(f"✓ Tokenizer loaded in {tokenizer_load_time:.2f}s")

        # Load model
        print("Loading model...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        if device == "cpu":
            model = model.to(device)
        model_load_time = time.time() - start_time
        print(f"✓ Model loaded in {model_load_time:.2f}s\n")

        if device == "cuda":
            print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB\n")

        # Run benchmarks
        results = []

        print("=" * 80)
        print("Running benchmarks...")
        print("=" * 80)

        for target_length in target_prompt_lengths:
            # Generate prompt
            prompt, actual_length = generate_prompt(target_length, tokenizer)

            print(f"\n[{len(results)+1}/{len(target_prompt_lengths)}] Testing prompt length: {actual_length} tokens")

            # Run inference twice (first for warmup, second for measurement)
            # Warmup
            _ = run_inference_with_timing(model, tokenizer, prompt, device, max_new_tokens=50)

            # Actual measurement
            metrics = run_inference_with_timing(model, tokenizer, prompt, device, max_new_tokens=50)

            results.append(metrics)

            print(f"  Latency: {metrics['latency_ms']:.2f} ms")
            print(f"  Tokens/sec: {metrics['tokens_per_sec']:.2f}")
            print(f"  Time to first token (approx): {metrics['time_to_first_token_ms']:.2f} ms")

        # Save results to CSV
        print("\n" + "=" * 80)
        print(f"Saving results to {output_file}...")
        print("=" * 80)

        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['prompt_length', 'latency_ms', 'tokens_per_sec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow({
                    'prompt_length': result['prompt_length'],
                    'latency_ms': result['latency_ms'],
                    'tokens_per_sec': result['tokens_per_sec']
                })

        print(f"✓ Results saved to {output_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Prompt Length':<15} {'Latency (ms)':<15} {'Tokens/sec':<15}")
        print("-" * 80)
        for result in results:
            print(f"{result['prompt_length']:<15} {result['latency_ms']:<15.2f} {result['tokens_per_sec']:<15.2f}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during inference: {type(e).__name__}")
        print(f"   {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
