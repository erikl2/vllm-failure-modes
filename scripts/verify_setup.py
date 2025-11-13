#!/usr/bin/env python3
"""
Lambda Labs A100 Setup Verification Script
Tests CUDA availability, model loading, and inference
"""

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_cuda():
    """Check CUDA availability and print GPU info"""
    print_section("1. CUDA & GPU Check")

    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   This script requires a GPU with CUDA support.")
        return False

    print("✅ CUDA is available")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {props.name}")
        print(f"   - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - Compute Capability: {props.major}.{props.minor}")

    return True


def load_model():
    """Load ungated Llama-like model"""
    print_section("2. Model Loading Test")

    # Using Mistral-7B-Instruct-v0.3 - ungated, similar architecture to Llama
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"   Loading: {model_name}")
    print(f"   Note: Using Mistral-7B (ungated, Llama-like architecture)")
    print(f"   Precision: torch.float16")
    print(f"   Device Map: auto\n")

    try:
        # Load tokenizer
        print("   [1/2] Loading tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - start_time
        print(f"   ✅ Tokenizer loaded in {tokenizer_time:.2f}s")

        # Load model
        print("\n   [2/2] Loading model...")
        start_time = time.time()

        # Reset GPU memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            use_safetensors=True,
        )

        load_time = time.time() - start_time

        # Get memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3

        print(f"   ✅ Model loaded in {load_time:.2f}s")
        print(f"\n   GPU Memory Usage:")
        print(f"   - Allocated: {memory_allocated:.2f} GB")
        print(f"   - Reserved: {memory_reserved:.2f} GB")
        print(f"   - Peak: {peak_memory:.2f} GB")

        return model, tokenizer, load_time

    except Exception as e:
        print(f"\n   ❌ Model loading failed!")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        return None, None, None


def run_test_inference(model, tokenizer):
    """Run a single test inference"""
    print_section("3. Test Inference")

    test_prompt = "Explain continuous batching"
    max_new_tokens = 50

    print(f"   Prompt: '{test_prompt}'")
    print(f"   Max new tokens: {max_new_tokens}\n")

    try:
        # Tokenize
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        print(f"   Input tokens: {input_length}")

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Run inference
        print("   Running inference...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        inference_time = time.time() - start_time

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_length = outputs.shape[1]
        tokens_generated = output_length - input_length

        # Get peak memory during inference
        peak_memory_inference = torch.cuda.max_memory_allocated(0) / 1024**3

        print(f"\n   ✅ Inference completed in {inference_time:.2f}s")
        print(f"\n   Results:")
        print(f"   - Tokens generated: {tokens_generated}")
        print(f"   - Tokens/second: {tokens_generated / inference_time:.2f}")
        print(f"   - Peak GPU memory: {peak_memory_inference:.2f} GB")

        print(f"\n   Generated text:")
        print(f"   {'-' * 76}")
        print(f"   {generated_text}")
        print(f"   {'-' * 76}")

        return inference_time, tokens_generated, peak_memory_inference

    except Exception as e:
        print(f"\n   ❌ Inference failed!")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        return None, None, None


def print_summary(load_time, inference_time, tokens_generated, peak_memory):
    """Print final summary"""
    print_section("SUMMARY")

    if all(v is not None for v in [load_time, inference_time, tokens_generated, peak_memory]):
        print("   ✅ ALL CHECKS PASSED")
        print(f"\n   Timing:")
        print(f"   - Model load time: {load_time:.2f}s")
        print(f"   - Inference time: {inference_time:.2f}s")
        print(f"   - Throughput: {tokens_generated / inference_time:.2f} tokens/sec")
        print(f"\n   Memory:")
        print(f"   - Peak GPU usage: {peak_memory:.2f} GB")
        print(f"\n   Status: Ready for benchmarking! 🚀")
        print("=" * 80 + "\n")
        return True
    else:
        print("   ❌ VERIFICATION FAILED")
        print("\n   Please check the errors above and ensure:")
        print("   - CUDA drivers are installed correctly")
        print("   - You have access to meta-llama/Meta-Llama-3-8B-Instruct on HuggingFace")
        print("   - You're authenticated with HuggingFace (huggingface-cli login)")
        print("=" * 80 + "\n")
        return False


def main():
    print("\n" + "=" * 80)
    print("  Lambda Labs A100 Setup Verification")
    print("=" * 80)

    # Check CUDA
    if not check_cuda():
        sys.exit(1)

    # Load model
    model, tokenizer, load_time = load_model()
    if model is None:
        sys.exit(1)

    # Run test inference
    inference_time, tokens_generated, peak_memory = run_test_inference(model, tokenizer)
    if inference_time is None:
        sys.exit(1)

    # Print summary
    success = print_summary(load_time, inference_time, tokens_generated, peak_memory)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
