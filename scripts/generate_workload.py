#!/usr/bin/env python3
"""
Generate realistic workload for LLM inference benchmarking.
Creates 100 prompts (70 short, 30 long) with Poisson arrival times.
"""

import json
import random
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not found. Install with: pip install tiktoken")
    exit(1)


# Realistic ML/AI topics for prompts
TOPICS = [
    "continuous batching",
    "GPU memory management",
    "transformer attention mechanisms",
    "ML inference optimization",
    "model quantization",
    "KV cache management",
    "prefix caching",
    "speculative decoding",
    "tensor parallelism",
    "pipeline parallelism",
    "dynamic batching",
    "CUDA kernel optimization",
    "mixed precision inference",
    "model serving architectures",
    "load balancing strategies",
    "request scheduling algorithms",
    "memory fragmentation",
    "PagedAttention",
    "flash attention",
    "rotary positional embeddings",
]

# Short prompt templates (target 50-200 tokens)
SHORT_TEMPLATES = {
    "questions": [
        "What is {topic}?",
        "Explain {topic} in one sentence.",
        "Define {topic} and give a brief example.",
        "How does {topic} work?",
        "What are the benefits of {topic}?",
        "Why is {topic} important for LLM inference?",
        "What is the difference between {topic1} and {topic2}?",
        "When should you use {topic}?",
        "What are common pitfalls with {topic}?",
        "How does {topic} improve performance?",
    ],
    "summaries": [
        "Summarize {topic} in 2-3 sentences.",
        "Give the key points about {topic}.",
        "Provide a brief overview of {topic}.",
        "What are the main concepts in {topic}?",
        "List the core principles of {topic}.",
    ],
    "quick_tasks": [
        "List 3 examples of {topic}.",
        "Compare {topic1} and {topic2} briefly.",
        "Name 5 advantages of {topic}.",
        "What are 3 challenges with {topic}?",
        "Give 4 use cases for {topic}.",
        "List the steps to implement {topic}.",
        "What metrics measure {topic} effectiveness?",
    ],
}

# Long prompt templates (target 500-1000 tokens)
LONG_TEMPLATES = {
    "essays": [
        "Write a detailed essay explaining {topic}, including its history, core mechanisms, advantages, disadvantages, and real-world applications in modern LLM serving systems.",
        "Provide a comprehensive explanation of {topic}. Discuss the technical implementation details, the problems it solves, how it compares to alternative approaches, and why it matters for production ML systems.",
        "Write an in-depth technical article about {topic}. Cover the fundamental concepts, mathematical foundations if applicable, implementation considerations, performance characteristics, and best practices for deployment.",
    ],
    "analyses": [
        "Analyze the relationship between {topic1} and {topic2}. Explain how they interact, their synergies and conflicts, implementation trade-offs, and provide specific examples of systems that use both effectively.",
        "Provide a detailed analysis of {topic} in the context of modern LLM inference. Discuss the technical challenges, current state-of-the-art solutions, performance implications, memory requirements, and future directions in this area.",
        "Critically analyze {topic}. What are its strengths and weaknesses? How does it perform under different workload conditions? What are the implementation challenges? Compare it with alternative approaches and explain when each is most appropriate.",
    ],
    "guides": [
        "Create a comprehensive guide for implementing {topic} in a production LLM serving system. Include step-by-step instructions, code considerations, common mistakes to avoid, performance tuning tips, and monitoring recommendations.",
        "Write a detailed tutorial on {topic}. Explain the concepts from first principles, provide practical examples, discuss configuration options, show how to benchmark and validate the implementation, and give troubleshooting advice.",
        "Develop a complete guide to understanding and implementing {topic}. Cover the theoretical background, practical implementation steps, integration with existing systems, performance optimization techniques, and production deployment considerations.",
    ],
    "stories": [
        "Write a detailed technical narrative about a team implementing {topic} in their LLM inference system. Describe the initial performance problems they faced, their investigation process, the solution design, implementation challenges, testing approach, and the final results with specific metrics.",
        "Create a story about debugging a production issue related to {topic}. Include detailed descriptions of the symptoms, the investigation methodology, tools used for diagnosis, the root cause analysis, the fix implementation, and lessons learned for preventing similar issues.",
        "Tell the story of how {topic} evolved in the field of LLM inference. Discuss the original problems that needed solving, early approaches and their limitations, breakthrough innovations, current best practices, and what the future might hold.",
    ],
}


def get_tokenizer():
    """Initialize tiktoken tokenizer with cl100k_base encoding."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using tiktoken."""
    return len(tokenizer.encode(text))


def generate_short_prompt(tokenizer) -> Tuple[str, int]:
    """
    Generate a short prompt (target 50-200 tokens).
    Returns: (prompt_text, token_count)
    """
    # Pick random category and template
    category = random.choice(list(SHORT_TEMPLATES.keys()))
    template = random.choice(SHORT_TEMPLATES[category])

    # Generate prompt
    if "{topic1}" in template and "{topic2}" in template:
        topic1, topic2 = random.sample(TOPICS, 2)
        prompt = template.format(topic1=topic1, topic2=topic2)
    else:
        topic = random.choice(TOPICS)
        prompt = template.format(topic=topic)

    tokens = count_tokens(prompt, tokenizer)

    # Keep adding context until we reach target range
    additions = [
        " Provide specific technical details and explain the underlying mechanisms.",
        " Include examples from real production systems and discuss their performance characteristics.",
        " Explain the technical implementation details and common best practices.",
        " Discuss the performance implications including latency, throughput, and memory usage.",
        " Compare with alternative approaches and explain when each is most appropriate.",
        " Describe the trade-offs involved and how to choose the right configuration.",
        " Explain how this integrates with other components in a typical ML inference system.",
        " Discuss common pitfalls and mistakes to avoid when implementing this in production.",
        " Include relevant metrics and benchmarks that demonstrate the performance benefits.",
        " Describe the historical context and why this approach was developed.",
    ]

    random.shuffle(additions)  # Randomize order

    for addition in additions:
        if tokens >= 50:
            break
        prompt += addition
        tokens = count_tokens(prompt, tokenizer)

    # If still too short (unlikely), add a comprehensive request
    if tokens < 50:
        prompt += " Please provide a comprehensive answer with concrete examples, technical details, and practical considerations for production deployment."
        tokens = count_tokens(prompt, tokenizer)

    return prompt, tokens


def generate_long_prompt(tokenizer) -> Tuple[str, int]:
    """
    Generate a long prompt (target 500-1000 tokens).
    Returns: (prompt_text, token_count)
    """
    # Pick random category and template
    category = random.choice(list(LONG_TEMPLATES.keys()))
    template = random.choice(LONG_TEMPLATES[category])

    # Generate base prompt
    if "{topic1}" in template and "{topic2}" in template:
        topic1, topic2 = random.sample(TOPICS, 2)
        prompt = template.format(topic1=topic1, topic2=topic2)
    else:
        topic = random.choice(TOPICS)
        prompt = template.format(topic=topic)

    tokens = count_tokens(prompt, tokenizer)

    # If too short, add more detailed requirements to reach target range
    expansions = [
        " Include specific code examples and implementation patterns.",
        " Discuss the mathematical foundations and algorithmic complexity.",
        " Provide benchmarking results and performance comparisons with actual numbers.",
        " Explain the hardware considerations including GPU architecture impacts.",
        " Cover edge cases, failure modes, and error handling strategies.",
        " Discuss scalability considerations for handling thousands of concurrent requests.",
        " Include memory management details and optimization techniques.",
        " Explain how this interacts with other components in the inference stack.",
        " Provide configuration recommendations for different hardware setups.",
        " Discuss monitoring, observability, and debugging approaches.",
    ]

    while tokens < 500:
        prompt += random.choice(expansions)
        tokens = count_tokens(prompt, tokenizer)

    # Make sure we don't exceed 1000 tokens (though our templates shouldn't)
    if tokens > 1000:
        # Trim the prompt (rare case)
        words = prompt.split()
        while tokens > 1000:
            words = words[:-10]  # Remove 10 words at a time
            prompt = " ".join(words)
            tokens = count_tokens(prompt, tokenizer)

    return prompt, tokens


def generate_poisson_arrivals(n: int, lambda_rate: float) -> List[float]:
    """
    Generate Poisson arrival times.

    Args:
        n: Number of arrivals
        lambda_rate: Average rate (requests per second)

    Returns:
        List of arrival times starting from 0.0
    """
    # Generate inter-arrival times from exponential distribution
    inter_arrivals = np.random.exponential(1.0 / lambda_rate, n)

    # Convert to cumulative arrival times
    arrival_times = np.cumsum(inter_arrivals)

    # Start from 0
    arrival_times = np.insert(arrival_times[:-1], 0, 0.0)

    return arrival_times.tolist()


def generate_workload(
    total_prompts: int = 100,
    short_count: int = 70,
    long_count: int = 30,
    lambda_rate: float = 10.0,
    seed: int = 42
) -> Dict:
    """
    Generate complete workload specification.

    Args:
        total_prompts: Total number of prompts
        short_count: Number of short prompts
        long_count: Number of long prompts
        lambda_rate: Poisson arrival rate (requests/second)
        seed: Random seed for reproducibility

    Returns:
        Workload dictionary with metadata and prompts
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating workload with {total_prompts} prompts...")
    print(f"  - {short_count} short prompts (50-200 tokens)")
    print(f"  - {long_count} long prompts (500-1000 tokens)")
    print(f"  - Arrival rate: λ={lambda_rate} req/s")
    print()

    # Initialize tokenizer
    print("Initializing tiktoken tokenizer (cl100k_base)...")
    tokenizer = get_tokenizer()

    # Generate prompts
    print("Generating prompts...")
    prompts = []

    # Generate short prompts
    print(f"  Generating {short_count} short prompts...")
    for i in range(short_count):
        prompt_text, tokens = generate_short_prompt(tokenizer)
        prompts.append({
            "id": len(prompts),
            "prompt": prompt_text,
            "target_tokens": 125,  # midpoint of 50-200
            "actual_tokens": tokens,
            "category": "short"
        })
        if (i + 1) % 20 == 0:
            print(f"    Generated {i + 1}/{short_count} short prompts")

    # Generate long prompts
    print(f"  Generating {long_count} long prompts...")
    for i in range(long_count):
        prompt_text, tokens = generate_long_prompt(tokenizer)
        prompts.append({
            "id": len(prompts),
            "prompt": prompt_text,
            "target_tokens": 750,  # midpoint of 500-1000
            "actual_tokens": tokens,
            "category": "long"
        })
        if (i + 1) % 10 == 0:
            print(f"    Generated {i + 1}/{long_count} long prompts")

    # Shuffle prompts to mix short and long
    random.shuffle(prompts)

    # Reassign IDs after shuffle
    for i, prompt in enumerate(prompts):
        prompt["id"] = i

    # Generate arrival times
    print("Generating Poisson arrival times...")
    arrival_times = generate_poisson_arrivals(total_prompts, lambda_rate)

    # Assign arrival times (already sorted)
    for prompt, arrival_time in zip(prompts, arrival_times):
        prompt["arrival_time"] = round(arrival_time, 6)

    # Sort by arrival time
    prompts.sort(key=lambda p: p["arrival_time"])

    # Create workload structure
    workload = {
        "metadata": {
            "total_prompts": total_prompts,
            "short_count": short_count,
            "long_count": long_count,
            "lambda_rate": lambda_rate,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": seed
        },
        "prompts": prompts
    }

    return workload


def verify_workload(workload: Dict) -> None:
    """Verify workload meets requirements and print summary."""
    prompts = workload["prompts"]

    # Count categories
    short_prompts = [p for p in prompts if p["category"] == "short"]
    long_prompts = [p for p in prompts if p["category"] == "long"]

    # Token statistics
    short_tokens = [p["actual_tokens"] for p in short_prompts]
    long_tokens = [p["actual_tokens"] for p in long_prompts]

    print("\n" + "="*60)
    print("WORKLOAD VERIFICATION")
    print("="*60)

    print(f"\nPrompt Distribution:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Short prompts: {len(short_prompts)}")
    print(f"  Long prompts: {len(long_prompts)}")

    print(f"\nShort Prompt Token Statistics:")
    print(f"  Range: {min(short_tokens)} - {max(short_tokens)} tokens")
    print(f"  Mean: {np.mean(short_tokens):.1f} tokens")
    print(f"  Median: {np.median(short_tokens):.1f} tokens")
    print(f"  Target range: 50-200 tokens")

    print(f"\nLong Prompt Token Statistics:")
    print(f"  Range: {min(long_tokens)} - {max(long_tokens)} tokens")
    print(f"  Mean: {np.mean(long_tokens):.1f} tokens")
    print(f"  Median: {np.median(long_tokens):.1f} tokens")
    print(f"  Target range: 500-1000 tokens")

    arrival_times = [p["arrival_time"] for p in prompts]
    print(f"\nArrival Times:")
    print(f"  Start: {min(arrival_times):.3f}s")
    print(f"  End: {max(arrival_times):.3f}s")
    print(f"  Duration: {max(arrival_times) - min(arrival_times):.3f}s")

    # Assertions
    print(f"\nAssertion Checks:")
    try:
        assert len(short_prompts) == 70, f"Expected 70 short prompts, got {len(short_prompts)}"
        print("  ✓ Short prompt count: 70")

        assert len(long_prompts) == 30, f"Expected 30 long prompts, got {len(long_prompts)}"
        print("  ✓ Long prompt count: 30")

        assert all(50 <= t <= 200 for t in short_tokens), "Short prompt tokens out of range"
        print("  ✓ Short prompt tokens: 50-200 range")

        assert all(500 <= t <= 1000 for t in long_tokens), "Long prompt tokens out of range"
        print("  ✓ Long prompt tokens: 500-1000 range")

        assert arrival_times[0] == 0.0, "First arrival time should be 0.0"
        print("  ✓ Arrival times start at 0.0")

        assert arrival_times == sorted(arrival_times), "Arrival times not sorted"
        print("  ✓ Arrival times sorted")

        print("\n✓ All assertions passed!")

    except AssertionError as e:
        print(f"\n✗ Assertion failed: {e}")
        raise

    print("="*60 + "\n")


def main():
    """Main execution function."""
    output_file = "workload.json"

    # Generate workload
    workload = generate_workload(
        total_prompts=100,
        short_count=70,
        long_count=30,
        lambda_rate=10.0,
        seed=42
    )

    # Verify workload
    verify_workload(workload)

    # Write to file
    print(f"Writing workload to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(workload, f, indent=2)

    print(f"✓ Workload generated successfully!")
    print(f"✓ Output written to: {output_file}")

    # Show a few example prompts
    print("\nExample prompts:")
    for i, prompt in enumerate(workload["prompts"][:3], 1):
        print(f"\n  [{i}] {prompt['category'].upper()} ({prompt['actual_tokens']} tokens)")
        print(f"      Arrival: {prompt['arrival_time']:.3f}s")
        preview = prompt['prompt'][:100] + "..." if len(prompt['prompt']) > 100 else prompt['prompt']
        print(f"      \"{preview}\"")


if __name__ == "__main__":
    main()
