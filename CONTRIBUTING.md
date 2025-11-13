# Contributing to vLLM Failure Modes Research

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing code, documentation, and experimental results.

## How to Contribute

### Reporting Issues

If you encounter bugs or have suggestions:

1. Check existing issues to avoid duplicates
2. Open a new issue with a clear title and description
3. Include:
   - Your hardware setup (GPU model, memory, etc.)
   - Software versions (vLLM, PyTorch, CUDA)
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior

### Extending the Research

We welcome contributions that extend this work:

#### Testing New Configurations

- **Different concurrency levels**: C=1, 2, 8, 16, 32
- **Different hardware**: H100, A10, L4, other GPUs
- **Different models**: Llama-70B, Mistral, GPT-J, etc.
- **Different workloads**: Code generation, technical docs, etc.

#### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-concurrency-test`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

### Experimental Methodology

When contributing new experimental results:

1. **Multi-seed testing**: Use at least 3 random seeds
2. **Sustained load**: Run tests for minimum 5 minutes
3. **Clean state**: Verify GPU memory cleanup between runs
4. **Documentation**: Record all parameters and configurations
5. **Reproducibility**: Provide detailed setup instructions

### Data Format

Experimental results should be in CSV format with columns:
- `gpu_memory`: GPU memory utilization setting
- `seed`: Random seed used
- `concurrency`: Concurrency level
- `status`: SUCCESS, CRASH, or FAILED_TO_START
- `p50_latency_ms`: 50th percentile latency
- `p90_latency_ms`: 90th percentile latency
- `p95_latency_ms`: 95th percentile latency
- `p99_latency_ms`: 99th percentile latency
- `throughput_tps`: Tokens per second
- `crash_time_s`: Time to crash (if applicable)

### Documentation

When adding new features or experiments:

1. Update README.md with new capabilities
2. Add docstrings to new functions
3. Include usage examples
4. Document any new dependencies

### Testing Infrastructure

- Test scripts should clean up GPU memory on exit
- Use health checks before starting benchmarks
- Implement timeouts for long-running operations
- Log all errors and crashes with timestamps

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/vllm-failure-modes.git
cd vllm-failure-modes

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Running Tests

```bash
# Run test suite (if available)
pytest tests/

# Check code style
black --check scripts/ analysis/ automation/
flake8 scripts/ analysis/ automation/

# Type checking
mypy scripts/ analysis/ automation/
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of changes
4. Reference any related issues
5. Wait for review from maintainers

## Research Ethics

When contributing to this research:

- Report results honestly, including negative findings
- Document limitations clearly
- Respect computational resource constraints
- Share data and methodologies openly
- Credit prior work appropriately

## Questions?

If you have questions about contributing, please open an issue with the "question" label.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
