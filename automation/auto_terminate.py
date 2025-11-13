#!/usr/bin/env python3
"""
Lambda Labs Auto-Terminate Script

Automatically terminates the current Lambda Labs instance after a specified
time period to prevent runaway costs.

Usage:
    export LAMBDA_API_KEY=your_api_key
    export WAIT_MINUTES=120  # Optional, defaults to 120
    nohup python auto_terminate.py > /tmp/auto_terminate.log 2>&1 &

Environment Variables:
    LAMBDA_API_KEY: Lambda Labs API key (required)
    WAIT_MINUTES: Minutes to wait before terminating (default: 120)
"""

import os
import sys
import time
import base64
import requests
from datetime import datetime, timedelta


# Configuration
LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"
IP_INFO_URL = "https://ipinfo.io/ip"
STATUS_LOG_INTERVAL_MINUTES = 30


def log(message):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_env_var(name, required=True, default=None):
    """Get environment variable with validation"""
    value = os.environ.get(name)

    if value is None:
        if required:
            log(f"ERROR: Environment variable {name} is not set")
            log(f"Please run: export {name}=your_value")
            sys.exit(1)
        return default

    return value


def get_basic_auth_header(api_key):
    """Create Basic Auth header from API key"""
    # Lambda API uses: base64(api_key + ":")
    credentials = f"{api_key}:"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def get_public_ip():
    """Get this instance's public IP address"""
    try:
        log("Getting public IP address...")
        response = requests.get(IP_INFO_URL, timeout=10)
        response.raise_for_status()
        ip = response.text.strip()
        log(f"✓ Public IP: {ip}")
        return ip
    except Exception as e:
        log(f"ERROR: Failed to get public IP: {e}")
        sys.exit(1)


def find_instance_id(api_key, public_ip):
    """Find Lambda Labs instance ID by matching public IP"""
    try:
        log("Querying Lambda Labs API for instance list...")

        headers = get_basic_auth_header(api_key)
        url = f"{LAMBDA_API_BASE}/instances"

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        if "data" not in data:
            log(f"ERROR: Unexpected API response format: {data}")
            sys.exit(1)

        instances = data["data"]
        log(f"Found {len(instances)} total instance(s)")

        # Find instance matching our public IP
        for instance in instances:
            instance_id = instance.get("id")
            instance_ip = instance.get("ip")
            instance_name = instance.get("name", "N/A")
            instance_status = instance.get("status", "N/A")

            log(f"  - Instance: {instance_name} (ID: {instance_id}, IP: {instance_ip}, Status: {instance_status})")

            if instance_ip == public_ip:
                log(f"✓ Found matching instance!")
                log(f"  ID: {instance_id}")
                log(f"  Name: {instance_name}")
                log(f"  Status: {instance_status}")
                return instance_id

        log(f"ERROR: No instance found with IP {public_ip}")
        log("Please verify:")
        log("  - This script is running on a Lambda Labs instance")
        log("  - The API key is correct")
        log("  - The instance is active")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            log("ERROR: Authentication failed - invalid API key")
        else:
            log(f"ERROR: HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"ERROR: Failed to query instances: {e}")
        sys.exit(1)


def terminate_instance(api_key, instance_id):
    """Terminate Lambda Labs instance"""
    try:
        log(f"Terminating instance {instance_id}...")

        headers = get_basic_auth_header(api_key)
        headers["Content-Type"] = "application/json"

        url = f"{LAMBDA_API_BASE}/instance-operations/terminate"
        payload = {"instance_ids": [instance_id]}

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        log(f"✓ Termination request successful!")
        log(f"Response: {data}")

        return True

    except requests.exceptions.HTTPError as e:
        log(f"ERROR: Failed to terminate instance: HTTP {e.response.status_code}")
        log(f"Response: {e.response.text}")
        return False
    except Exception as e:
        log(f"ERROR: Failed to terminate instance: {e}")
        return False


def format_time_remaining(minutes):
    """Format minutes into human-readable time"""
    hours = minutes // 60
    mins = minutes % 60

    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def wait_with_status_updates(wait_minutes):
    """Wait for specified minutes, logging status every 30 minutes"""
    end_time = datetime.now() + timedelta(minutes=wait_minutes)
    next_status_time = datetime.now() + timedelta(minutes=STATUS_LOG_INTERVAL_MINUTES)

    log(f"Waiting {wait_minutes} minutes until {end_time.strftime('%Y-%m-%d %H:%M:%S')}...")
    log(f"Status updates every {STATUS_LOG_INTERVAL_MINUTES} minutes")
    log("")

    while datetime.now() < end_time:
        # Check if it's time for a status update
        if datetime.now() >= next_status_time:
            remaining = (end_time - datetime.now()).total_seconds() / 60
            remaining_minutes = int(remaining)

            log(f"⏰ Status: {format_time_remaining(remaining_minutes)} remaining until termination")
            log(f"   Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log(f"   Termination time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

            next_status_time = datetime.now() + timedelta(minutes=STATUS_LOG_INTERVAL_MINUTES)

        # Sleep for 1 minute at a time to avoid long blocking
        time.sleep(60)

    log("")
    log(f"⏰ Wait period completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    log("=" * 80)
    log("Lambda Labs Auto-Terminate Script")
    log("=" * 80)
    log("")

    # Get configuration from environment
    api_key = get_env_var("LAMBDA_API_KEY", required=True)
    wait_minutes = int(get_env_var("WAIT_MINUTES", required=False, default=120))

    log(f"Configuration:")
    log(f"  API Key: {api_key[:15]}... (masked)")
    log(f"  Wait time: {wait_minutes} minutes ({format_time_remaining(wait_minutes)})")
    log("")

    # Get public IP
    public_ip = get_public_ip()
    log("")

    # Find instance ID
    instance_id = find_instance_id(api_key, public_ip)
    log("")

    # Calculate termination time
    termination_time = datetime.now() + timedelta(minutes=wait_minutes)

    log("=" * 80)
    log(f"⚠️  INSTANCE WILL BE TERMINATED IN {wait_minutes} MINUTES")
    log(f"   Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"   Termination time: {termination_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"   Instance ID: {instance_id}")
    log(f"   Public IP: {public_ip}")
    log("=" * 80)
    log("")

    # Wait for specified time
    wait_with_status_updates(wait_minutes)

    # Terminate instance
    log("=" * 80)
    log("INITIATING TERMINATION")
    log("=" * 80)
    log("")

    success = terminate_instance(api_key, instance_id)

    if success:
        log("")
        log("=" * 80)
        log("✓ Instance termination completed successfully")
        log(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("=" * 80)
        sys.exit(0)
    else:
        log("")
        log("=" * 80)
        log("✗ Instance termination failed")
        log("  Please terminate manually via Lambda Labs web console")
        log("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("")
        log("=" * 80)
        log("⚠️  Script interrupted by user (Ctrl+C)")
        log("   Instance will NOT be terminated")
        log("=" * 80)
        sys.exit(130)
    except Exception as e:
        log("")
        log("=" * 80)
        log(f"✗ FATAL ERROR: {e}")
        log("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
