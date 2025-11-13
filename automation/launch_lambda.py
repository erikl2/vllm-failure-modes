#!/usr/bin/env python3
"""
Launch a Lambda Labs GPU instance.
"""

import os
import sys
import json
import time
import requests

def launch_instance(api_key, region_name="us-west-1", instance_type_name="gpu_1x_a100"):
    """Launch a Lambda Labs instance."""

    base_url = "https://cloud.lambdalabs.com/api/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Get SSH key ID
    print("Fetching SSH keys...")
    response = requests.get(f"{base_url}/ssh-keys", headers=headers)
    response.raise_for_status()
    ssh_keys = response.json().get("data", [])

    if not ssh_keys:
        print("✗ No SSH keys found in your Lambda Labs account")
        print("Please add an SSH key at https://cloud.lambdalabs.com/ssh-keys")
        return None

    ssh_key_ids = [key["id"] for key in ssh_keys]
    print(f"Found {len(ssh_key_ids)} SSH key(s)")

    # Launch instance
    print(f"\nLaunching {instance_type_name} in {region_name}...")

    payload = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": [key["name"] for key in ssh_keys],
        "file_system_names": [],
        "quantity": 1
    }

    response = requests.post(
        f"{base_url}/instance-operations/launch",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        instances = result.get("data", {}).get("instance_ids", [])

        if instances:
            instance_id = instances[0]
            print(f"✓ Instance launched: {instance_id}")

            # Wait for instance to get an IP
            print("Waiting for instance to be ready...")
            for i in range(60):
                time.sleep(5)

                response = requests.get(f"{base_url}/instances", headers=headers)
                response.raise_for_status()
                all_instances = response.json().get("data", [])

                for inst in all_instances:
                    if inst["id"] == instance_id:
                        ip = inst.get("ip")
                        status = inst.get("status")

                        print(f"  Status: {status}, IP: {ip or 'pending'}")

                        if ip and status == "active":
                            print(f"\n✓ Instance ready!")
                            print(f"  ID: {instance_id}")
                            print(f"  IP: {ip}")
                            print(f"  Type: {instance_type_name}")
                            return {"id": instance_id, "ip": ip}

            print("✗ Instance did not become ready within 5 minutes")
            return None
        else:
            print("✗ No instance ID returned")
            return None
    else:
        error_msg = response.json().get("error", {}).get("message", response.text)
        print(f"✗ Failed to launch instance: {error_msg}")
        return None

if __name__ == "__main__":
    api_key = os.getenv("LAMBDA_API_KEY")
    if not api_key:
        print("✗ LAMBDA_API_KEY not set in environment")
        sys.exit(1)

    # Try multiple regions
    regions = ["us-west-1", "us-west-2", "us-east-1", "us-south-1", "us-midwest-1"]

    for region in regions:
        print(f"\n{'='*60}")
        print(f"Trying region: {region}")
        print('='*60)
        result = launch_instance(api_key, region_name=region)

        if result:
            print(f"\n✓ SUCCESS! Instance launched in {region}")
            print(f"\nTo use this instance, update your .env file:")
            print(f'export LAMBDA_HOST="ubuntu@{result["ip"]}"')
            sys.exit(0)
        else:
            print(f"No capacity in {region}, trying next region...")

    print("\n✗ No capacity available in any region")
    print("Please try again later or launch manually at https://cloud.lambdalabs.com/instances")
    sys.exit(1)
