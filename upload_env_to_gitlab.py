#!/usr/bin/env python3
"""
Simple GitLab .env uploader - minimal version
"""
import os, re, base64, requests, click
from dotenv import load_dotenv

load_dotenv(override = True)

#On GitLab:
#1. Go to Access Tokens and create a new API key
#2. Set the following permissions: api, read_api, read_repository, write_repository
#3. Copy the token and paste on GITLAB_TOKEN

GITLAB_URL = os.environ["GITLAB_URL"]
PROJECT_ID = os.environ["PROJECT_ID"]
GITLAB_TOKEN = os.environ["GITLAB_TOKEN"]

@click.command()
@click.option("--filename",
              type = str,
              default = ".env",
              help = "Secrets file to upload to GitLab")
def upload_env_to_gitlab(filename):
    api_url = f"{GITLAB_URL.rstrip('/')}/api/v4/projects/{PROJECT_ID}/variables"
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN, "Content-Type": "application/json"}
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Debug output
            print(f"Processing line {line_num}: {repr(line)}")
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                print(f"   → Skipping (empty or comment)")
                continue
            for var in ["GITLAB_TOKEN=", "GITLAB_URL=", "PROJECT_ID="]:
                contains_gitlab_variable = False
                if var in line:
                    contains_gitlab_variable = True
                    print(f"   → Skipping (contains {var})")
                    break
            if contains_gitlab_variable:
                continue
            # Check for = sign
            if '=' not in line:
                print(f"   → Skipping (no equals sign)")
                continue
            # Split key and value
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            # Remove quotes from value (both single and double)
            if len(value) >= 2:
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
            print(f"   → Key: {key}")
            print(f"   → Value: {value[:50]}..." if len(value) > 50 else f"   → Value: {value}")
            # Always encode to base64 (since you're doing that anyway)
            actual_key = key
            actual_value = base64.b64encode(value.encode()).decode()
            print(f"🔒 {actual_key} (base64 encoded)")
            # Here's where you'd do your GitLab upload
            # upload_to_gitlab(actual_key, actual_value)
            print()
            # Upload to GitLab
            data = {
                "key": actual_key, 
                "value": actual_value, 
                "masked": True, 
                "protected": False} #to avoid conflicts with global variables outside repo
            response = requests.post(
                api_url, 
                headers = headers, 
                json = data)
            if response.status_code == 201:
                print(f"✅ Uploaded: {actual_key}")
            elif response.status_code == 400:
                # Try updating existing
                print(f"⚠️  Variable {actual_key} already exists, trying to update...")
                update_response = requests.put(
                    f"{api_url}/{actual_key}",
                    headers = headers,
                    json = data
                )
                # Check the update response status
                if update_response.status_code == 200:
                    print(f"🔄 Updated: {actual_key}")
                else:
                    print(f"❌ Failed to update {actual_key} - Status: {update_response.status_code}")
                    try:
                        error_msg = update_response.json().get('message', 'Unknown error')
                        print(f"   Error: {error_msg}")
                    except:
                        print(f"   Response: {update_response.text}")
            else:
                print(f"❌ Failed: {actual_key} - Status: {response.status_code}")
                try:
                    error_msg = response.json().get('message', 'Unknown error')
                    print(f"   Error: {error_msg}")
                except:
                    print(f"   Response: {response.text}")
            print()

if __name__ == "__main__":
    upload_env_to_gitlab()