# Catastrophic Failure Modes in vLLM: GPU Memory Threshold Study

This repository contains the experimental code and results for the research paper **"Catastrophic Failure Modes in vLLM: A Case Study of GPU Memory Thresholds at C=4 on A100 Hardware"** (November 2025).

## Abstract

Large language model (LLM) serving infrastructure must balance throughput maximization against system reliability. This case study empirically characterizes catastrophic failure modes in vLLM when GPU memory utilization falls below critical thresholds. Through 36 controlled experiments across six GPU memory settings (0.70–0.95) with concurrency C=4 on A100 40GB hardware running Llama-3.1-8B-Instruct, we establish that GPU memory utilization below 0.85 results in catastrophic crashes within 30–90 seconds.

## Key Findings

1. **Sharp Reliability Threshold**: GPU memory utilization of 0.85 is a critical threshold for C=4 concurrency. Configurations at or above this threshold achieve 100% reliability across multiple random seed initializations, while those below exhibit 0–33% reliability.

2. **Binary Failure Mode**: Unlike traditional web services that degrade gracefully, vLLM exhibits binary failure behavior—systems crash completely without warning signals.

3. **Seed-Dependent Variance**: Random seed initialization affects crash probability in the unstable region, creating production risk where validated configurations may fail with different seeds.

4. **Economic Implications**: Conservative GPU memory allocation (0.85–0.90) optimizes total cost of ownership when operational overhead and customer impact are factored in.

## Repository Structure

```
.
├── README.md                 # This file
├── docs/                     # Research paper and documentation
│   └── vllm_failure_modes_paper_v2_2_1.pdf
├── scripts/                  # Benchmarking and deployment scripts
│   ├── benchmark_vllm.py    # Main vLLM benchmarking script
│   ├── vllm_runner.py       # vLLM server runner
│   ├── generate_workload.py # Workload generation from ShareGPT
│   └── run_*.sh             # Experiment automation scripts
├── analysis/                 # Data analysis and plotting scripts
│   ├── analyze_threshold.py
│   ├── analyze_headroom.py
│   ├── generate_plots.py
│   └── analyze_and_plot.py
├── automation/               # Infrastructure management
│   ├── auto_cleanup.sh      # GPU cleanup automation
│   ├── health_check.py      # Server health monitoring
│   ├── lambda_api.py        # Lambda Labs API integration
│   └── launch_lambda.py     # Automated instance launch
├── results/                  # Experimental results (CSV files)
├── workload.json            # ShareGPT v3 dataset sample
└── requirements.txt         # Python dependencies
```

## Experimental Setup

### Hardware
- **GPU**: A100 40GB SXM4
- **Platform**: Lambda Labs cloud infrastructure
- **CPU**: 30 cores (AMD EPYC 7763)
- **RAM**: 200GB

### Software Stack
- **vLLM**: 0.11.0
- **PyTorch**: 2.0.1
- **CUDA**: 12.1
- **Python**: 3.10
- **Model**: Llama-3.1-8B-Instruct
- **Dataset**: ShareGPT v3

### Experimental Design
- **GPU memory settings**: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
- **Random seeds**: 42, 43, 44 (3 seeds per setting)
- **Concurrency level**: C=4 (fixed)
- **Test duration**: 5 minutes sustained load per experiment
- **Total experiments**: 36 (6 settings × 3 seeds × 2 with retries)

## Quick Start

### Prerequisites

```bash
# Python 3.10+
# CUDA 12.1+
# Access to A100 GPU (local or cloud)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vllm-failure-modes.git
cd vllm-failure-modes

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

#### Single GPU Memory Setting Test

```bash
# Test with specific GPU memory utilization and seed
cd scripts
python benchmark_vllm.py \
    --gpu-memory-utilization 0.85 \
    --seed 42 \
    --concurrency 4 \
    --duration 300
```

#### Full Threshold Characterization

```bash
# Run complete experiment suite (requires GPU access)
cd scripts
./run_threshold_validation.sh
```

#### Analyze Results

```bash
cd analysis
python analyze_threshold.py --results-dir ../results/
python generate_plots.py --results-dir ../results/
```

## Key Results

### Reliability by GPU Memory Setting (C=4)

| GPU Memory | Seed 42 | Seed 43 | Seed 44 | Success Rate | Mean Latency (P50) |
|------------|---------|---------|---------|--------------|---------------------|
| 0.70       | CRASH   | CRASH   | CRASH   | 0%           | N/A                 |
| 0.75       | CRASH   | SUCCESS | CRASH   | 33%          | 492ms*              |
| 0.80       | SUCCESS | CRASH   | CRASH   | 33%          | 478ms*              |
| **0.85**   | SUCCESS | SUCCESS | SUCCESS | **100%**     | **485ms**           |
| 0.90       | SUCCESS | SUCCESS | SUCCESS | 100%         | 490ms               |
| 0.95       | SUCCESS | SUCCESS | SUCCESS | 100%         | 498ms               |

*Mean calculated only from successful runs

### Crash Timing Characteristics

- **GPU=0.70**: Mean crash time 31s (range: 28–35s)
- **GPU=0.75**: Mean crash time 45s (range: 38–52s)
- **GPU=0.80**: Mean crash time 54s (range: 47–62s)

All crashes occurred within 30–90 seconds of sustained load, with CUDA Out of Memory errors as the primary failure mode.

## Practical Recommendations

### For Production Deployments

1. **Minimum Safe Threshold**: Set GPU memory utilization to **0.85 or higher** for C=4 workloads on A100 40GB hardware.

2. **Recommended Setting**: Use **0.90** (vLLM default) for optimal balance of throughput and safety margin.

3. **Multi-Seed Validation**: Test each configuration with at least 3 different random seeds to detect seed-dependent failure modes.

4. **Sustained Load Testing**: Conduct load tests for minimum 5 minutes to detect delayed crash modes.

5. **Automated Cleanup**: Implement automated GPU memory cleanup mechanisms (5–8% of experiments exhibited memory leaks).

### Cost-Benefit Analysis

Conservative GPU memory settings (0.85–0.90) minimize total cost of ownership:

- **Aggressive (GPU=0.80)**: $29,530/month (requires 3× overprovisioning due to 67% failure rate)
- **Recommended (GPU=0.87)**: $11,145/month (100% reliability, optimal throughput)
- **Conservative (GPU=0.95)**: $10,195/month (lowest TCO, maximum safety margin)

## Citation

If you use this work in your research, please cite:

```bibtex
@article{vllm-failure-modes-2025,
  title={Catastrophic Failure Modes in vLLM: A Case Study of GPU Memory Thresholds at C=4 on A100 Hardware},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025},
  month={November}
}
```

## Limitations

- **Concurrency scope**: Experiments focused exclusively on C=4 due to budget constraints
- **Hardware**: Testing limited to A100 40GB; other GPUs may exhibit different thresholds
- **Model**: Results specific to Llama-3.1-8B-Instruct
- **Workload**: ShareGPT v3 conversational data; other workloads may differ
- **Duration**: 5-minute tests may not capture long-running stability issues

## Future Work

- Characterize GPU memory thresholds across different concurrency levels (C=1, 2, 8, 16, 32)
- Test on different hardware (H100, A10, L4)
- Evaluate other models (70B, different architectures)
- Investigate root cause of catastrophic failures
- Develop admission control mechanisms for safe operation at lower GPU memory settings

## Acknowledgments

We thank Lambda Labs for providing A100 GPU access that made these experiments possible. We also acknowledge the vLLM development team for building an open-source serving framework that enables systematic investigation of production deployment challenges.

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

For questions about the research or code, please open an issue on GitHub.

---

**Note**: This repository contains research code for experimental characterization. Production deployments should implement additional safety mechanisms, monitoring, and testing beyond what is demonstrated here.
