#!/bin/bash
################################################################################
# Quick Deploy Script for Breaking Point Ablation
# Automates the deployment process to Lambda Labs
################################################################################

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  BREAKING POINT ABLATION - QUICK DEPLOY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Load environment variables
echo "▶ Loading environment variables..."
source .env
echo "  ✓ Lambda Host: $LAMBDA_HOST"
echo ""

# Step 1: Generate workload if it doesn't exist
if [ ! -f "workload.json" ]; then
    echo "▶ workload.json not found - generating..."
    echo ""

    # Check if tiktoken is installed
    if ! python -c "import tiktoken" 2>/dev/null; then
        echo "  Installing tiktoken..."
        pip install tiktoken
    fi

    python generate_workload.py
    echo ""
    echo "  ✓ workload.json generated"
else
    echo "✓ workload.json already exists"
fi
echo ""

# Step 2: Test SSH connection
echo "▶ Testing SSH connection..."
if ssh -i $LAMBDA_SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no $LAMBDA_HOST "echo 'Connected'" >/dev/null 2>&1; then
    echo "  ✓ SSH connection successful"
else
    echo "  ✗ Cannot connect to $LAMBDA_HOST"
    echo "  Please check:"
    echo "    - Instance is running"
    echo "    - LAMBDA_HOST in .env is correct"
    echo "    - SSH key exists: $LAMBDA_SSH_KEY"
    exit 1
fi
echo ""

# Step 3: Upload files
echo "▶ Uploading files to Lambda..."
SCP_OPTS="-i $LAMBDA_SSH_KEY -o StrictHostKeyChecking=no"

scp $SCP_OPTS benchmark_vllm.py $LAMBDA_HOST:~/
echo "  ✓ benchmark_vllm.py"

scp $SCP_OPTS run_breaking_point_ablation.sh $LAMBDA_HOST:~/
echo "  ✓ run_breaking_point_ablation.sh"

scp $SCP_OPTS workload.json $LAMBDA_HOST:~/
echo "  ✓ workload.json"

scp $SCP_OPTS requirements_complete.txt $LAMBDA_HOST:~/
echo "  ✓ requirements_complete.txt"

scp $SCP_OPTS analyze_breaking_point.py $LAMBDA_HOST:~/
echo "  ✓ analyze_breaking_point.py"

scp $SCP_OPTS generate_plots.py $LAMBDA_HOST:~/
echo "  ✓ generate_plots.py"

echo ""
echo "  ✓ All files uploaded"
echo ""

# Step 4: Create results directory and make script executable
echo "▶ Setting up remote environment..."
ssh $SCP_OPTS $LAMBDA_HOST "mkdir -p results && chmod +x run_breaking_point_ablation.sh"
echo "  ✓ Results directory created"
echo "  ✓ Script made executable"
echo ""

# Step 5: Install dependencies
echo "▶ Installing dependencies on remote instance..."
echo "  This will take ~15 minutes. Please be patient..."
echo ""

ssh $SCP_OPTS $LAMBDA_HOST bash <<'REMOTE_SCRIPT'
echo "Installing Python packages..."
pip install -q -r requirements_complete.txt

echo ""
echo "Verifying installations..."
python -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'  ✓ Transformers: {transformers.__version__}')"
python -c "import vllm; print('  ✓ vLLM installed')"
python -c "import aiohttp; print('  ✓ aiohttp installed')"
python -c "import tiktoken; print('  ✓ tiktoken installed')"
echo ""
echo "✓ All dependencies installed"
REMOTE_SCRIPT

echo ""

# Step 6: HuggingFace login
echo "▶ Logging in to HuggingFace..."
ssh $SCP_OPTS $LAMBDA_HOST "huggingface-cli login --token $HF_TOKEN" >/dev/null 2>&1
echo "  ✓ HuggingFace login successful"
echo ""

# All done!
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  ✓ DEPLOYMENT COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. SSH into Lambda and start the experiment in tmux:"
echo ""
echo "   ssh -i \$LAMBDA_SSH_KEY \$LAMBDA_HOST"
echo "   tmux new -s ablation"
echo "   ./run_breaking_point_ablation.sh"
echo ""
echo "2. When prompted, type 'y' to start"
echo ""
echo "3. Detach from tmux: Ctrl+B, then D"
echo ""
echo "4. Monitor progress from your local machine:"
echo ""
echo "   ./check_progress.sh"
echo ""
echo "Estimated runtime: 6-8 hours"
echo "Estimated cost: ~\$5"
echo ""
echo "Good luck! 🚀"
echo ""
