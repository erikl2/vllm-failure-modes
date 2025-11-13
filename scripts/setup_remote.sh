#!/bin/bash
set -e  # Exit on any error

#################################################################################
# Lambda Labs Remote Setup Script
#
# Sets up a Lambda Labs GPU instance for LLM benchmarking
#
# REQUIRED ENVIRONMENT VARIABLES:
#   - LAMBDA_HOST: SSH hostname (e.g., user@ip-address)
#   - LAMBDA_SSH_KEY: Path to SSH private key
#   - HF_TOKEN: HuggingFace API token
#   - LAMBDA_API_KEY: Lambda Labs API key
#
# USAGE:
#   export LAMBDA_HOST="ubuntu@123.45.67.89"
#   export LAMBDA_SSH_KEY="~/.ssh/lambda_key"
#   export HF_TOKEN="hf_..."
#   export LAMBDA_API_KEY="..."
#   bash setup_remote.sh
#################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_step() {
    echo -e "\n${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check required environment variables
check_env_vars() {
    print_step "Checking environment variables..."

    local missing_vars=()

    if [ -z "$LAMBDA_HOST" ]; then
        missing_vars+=("LAMBDA_HOST")
    fi

    if [ -z "$LAMBDA_SSH_KEY" ]; then
        missing_vars+=("LAMBDA_SSH_KEY")
    fi

    if [ -z "$HF_TOKEN" ]; then
        missing_vars+=("HF_TOKEN")
    fi

    if [ -z "$LAMBDA_API_KEY" ]; then
        missing_vars+=("LAMBDA_API_KEY")
    fi

    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Please set them before running this script:"
        echo "  export LAMBDA_HOST=\"ubuntu@your-ip\""
        echo "  export LAMBDA_SSH_KEY=\"~/.ssh/your_key\""
        echo "  export HF_TOKEN=\"hf_...\""
        echo "  export LAMBDA_API_KEY=\"...\""
        exit 1
    fi

    # Expand SSH key path
    LAMBDA_SSH_KEY="${LAMBDA_SSH_KEY/#\~/$HOME}"

    if [ ! -f "$LAMBDA_SSH_KEY" ]; then
        print_error "SSH key not found: $LAMBDA_SSH_KEY"
        exit 1
    fi

    print_success "Environment variables verified"
    echo "  Host: $LAMBDA_HOST"
    echo "  SSH Key: $LAMBDA_SSH_KEY"
    echo "  HF Token: ${HF_TOKEN:0:10}..."
    echo "  Lambda API Key: ${LAMBDA_API_KEY:0:10}..."
}

# Test SSH connection
test_ssh() {
    print_step "Testing SSH connection..."

    if ssh -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$LAMBDA_HOST" "echo 'Connection successful'" &>/dev/null; then
        print_success "SSH connection successful"
    else
        print_error "Cannot connect to $LAMBDA_HOST"
        echo "  Please check:"
        echo "  - Instance is running"
        echo "  - LAMBDA_HOST is correct"
        echo "  - SSH key has correct permissions (chmod 600 $LAMBDA_SSH_KEY)"
        exit 1
    fi
}

# Upload verify_setup.py
upload_files() {
    print_step "Uploading verify_setup.py to remote instance..."

    if [ ! -f "verify_setup.py" ]; then
        print_error "verify_setup.py not found in current directory"
        exit 1
    fi

    scp -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=no verify_setup.py "$LAMBDA_HOST:~/"
    print_success "File uploaded"
}

# Run remote setup commands
run_remote_setup() {
    print_step "Running setup on remote instance..."

    ssh -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=no "$LAMBDA_HOST" bash -s <<'REMOTE_SCRIPT'
set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Remote Setup Starting"
echo "════════════════════════════════════════════════════════════════════════════════"

# Update system packages
echo ""
echo "▶ Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
echo "✓ System updated"

# Install uv package manager
echo ""
echo "▶ Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ uv installed"
else
    echo "✓ uv already installed"
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Create Python virtual environment
echo ""
echo "▶ Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    uv venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate

# Install Python packages
echo ""
echo "▶ Installing Python packages..."
echo "  This may take several minutes for PyTorch..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers accelerate vllm huggingface_hub
echo "✓ Packages installed"

# Verify installations
echo ""
echo "▶ Verifying installations..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
echo "✓ Installations verified"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Remote Setup Complete"
echo "════════════════════════════════════════════════════════════════════════════════"
REMOTE_SCRIPT

    print_success "Remote setup completed"
}

# Login to HuggingFace and run verification
run_verification() {
    print_step "Logging in to HuggingFace and running verification..."

    ssh -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=no "$LAMBDA_HOST" HF_TOKEN="$HF_TOKEN" bash -s <<'REMOTE_SCRIPT'
set -e

# Activate venv
source venv/bin/activate

# Login to HuggingFace
echo ""
echo "▶ Logging in to HuggingFace..."
echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
echo "✓ HuggingFace login successful"

# Run verify_setup.py
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Running verify_setup.py"
echo "════════════════════════════════════════════════════════════════════════════════"
python verify_setup.py
REMOTE_SCRIPT

    print_success "Verification completed"
}

# Ask user what to do next
ask_next_step() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    print_success "Setup complete!"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Continue working on the instance"
    echo "  2. Terminate the instance (to avoid costs)"
    echo ""
    read -p "What would you like to do? (continue/terminate): " choice

    case "$choice" in
        terminate|t)
            print_warning "To terminate the instance, use the Lambda Labs web console or API"
            echo "  Instance: $LAMBDA_HOST"
            echo "  API Key: ${LAMBDA_API_KEY:0:10}..."
            ;;
        continue|c|*)
            print_success "Instance is ready for use!"
            echo ""
            echo "To SSH into the instance:"
            echo "  ssh -i $LAMBDA_SSH_KEY $LAMBDA_HOST"
            echo ""
            echo "To activate the Python environment:"
            echo "  source venv/bin/activate"
            ;;
    esac
}

# Main execution
main() {
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Lambda Labs Remote Setup"
    echo "════════════════════════════════════════════════════════════════════════════════"

    check_env_vars
    test_ssh
    upload_files
    run_remote_setup
    run_verification
    ask_next_step

    echo ""
    print_success "All done! 🚀"
}

# Run main function
main
