#!/usr/bin/env python3
"""
Lambda Labs API client for reliable instance management.
Replaces unreliable SSH shutdown commands.
"""

import os
import sys
import json
import time
import requests
from typing import Optional, Dict, List

class LambdaLabsAPI:
    """Client for Lambda Labs API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('LAMBDA_API_KEY')
        if not self.api_key:
            raise ValueError("LAMBDA_API_KEY not set in environment")

        self.base_url = "https://cloud.lambdalabs.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def list_instances(self) -> List[Dict]:
        """List all instances."""
        response = requests.get(
            f"{self.base_url}/instances",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json().get("data", [])

    def get_instance_by_ip(self, ip: str) -> Optional[Dict]:
        """Find instance by IP address."""
        instances = self.list_instances()
        for instance in instances:
            if instance.get("ip") == ip:
                return instance
        return None

    def get_instance_status(self, instance_id: str) -> Optional[str]:
        """Get instance status."""
        try:
            response = requests.get(
                f"{self.base_url}/instances/{instance_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json().get("data", {}).get("status")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return "not_found"
            raise

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance via API."""
        print(f"Terminating instance {instance_id} via Lambda Labs API...")

        response = requests.post(
            f"{self.base_url}/instance-operations/terminate",
            headers=self.headers,
            json={"instance_ids": [instance_id]}
        )

        if response.status_code in (200, 202):
            print(f"✓ Termination request accepted")
            return True
        else:
            print(f"✗ Termination failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    def verify_termination(self, instance_id: str, max_wait: int = 60) -> bool:
        """Verify instance actually terminated."""
        print(f"Verifying termination of instance {instance_id}...")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.get_instance_status(instance_id)

            if status in ("terminated", "not_found"):
                print(f"✓ Confirmed: Instance {instance_id} is {status}")
                return True
            elif status == "terminating":
                print(f"  Instance is terminating... (waiting)")
                time.sleep(5)
            else:
                print(f"  Instance status: {status}")
                time.sleep(5)

        print(f"✗ Warning: Could not verify termination after {max_wait}s")
        return False

    def terminate_and_verify(self, instance_id: str) -> bool:
        """Terminate instance and verify it actually stopped."""
        if self.terminate_instance(instance_id):
            return self.verify_termination(instance_id)
        return False


def terminate_by_ip(ip_address: str) -> bool:
    """Terminate a Lambda instance by its IP address."""
    api = LambdaLabsAPI()

    # Find instance by IP
    instance = api.get_instance_by_ip(ip_address)
    if not instance:
        print(f"✗ No instance found with IP {ip_address}")
        return False

    instance_id = instance["id"]
    print(f"Found instance {instance_id} at {ip_address}")

    # Terminate and verify
    return api.terminate_and_verify(instance_id)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lambda_api.py <instance_ip>")
        print("   or: python lambda_api.py list")
        sys.exit(1)

    if sys.argv[1] == "list":
        api = LambdaLabsAPI()
        instances = api.list_instances()
        print(f"\n{'ID':<20} {'IP':<16} {'Status':<15} {'Type':<15}")
        print("-" * 70)
        for inst in instances:
            print(f"{inst['id']:<20} {inst.get('ip', 'N/A'):<16} {inst.get('status', 'N/A'):<15} {inst.get('instance_type', {}).get('name', 'N/A'):<15}")
        sys.exit(0)

    ip = sys.argv[1]
    success = terminate_by_ip(ip)
    sys.exit(0 if success else 1)
