#!/bin/bash
# Upload files to Lambda instance

# Source environment variables
source .env

echo "Uploading files to Lambda..."
echo "Host: $LAMBDA_HOST"
echo ""

# Upload files (disable strict host key checking for Lambda instances)
SCP_OPTS="-i $LAMBDA_SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "Uploading generate_workload.py..."
scp $SCP_OPTS generate_workload.py $LAMBDA_HOST:~/

echo "Uploading benchmark_naive.py..."
scp $SCP_OPTS benchmark_naive.py $LAMBDA_HOST:~/

echo "Uploading analyze_results.py..."
scp $SCP_OPTS analyze_results.py $LAMBDA_HOST:~/

echo "Uploading deploy_checklist.txt..."
scp $SCP_OPTS deploy_checklist.txt $LAMBDA_HOST:~/

echo ""
echo "✓ All files uploaded successfully!"
echo ""
echo "Next steps:"
echo "1. SSH into Lambda: ssh -i \$LAMBDA_SSH_KEY \$LAMBDA_HOST"
echo "2. Check files: ls -lh *.py *.txt"
echo "3. Continue with deployment checklist"
