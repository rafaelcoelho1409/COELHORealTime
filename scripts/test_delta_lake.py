#!/usr/bin/env python3
"""
Delta Lake Table Test Script

Tests connectivity and reads data from Delta Lake tables on MinIO.
Run this script to verify that Spark Streaming is writing data correctly.

Usage:
    python scripts/test_delta_lake.py
    python scripts/test_delta_lake.py --table tfd
    python scripts/test_delta_lake.py --table eta
    python scripts/test_delta_lake.py --table ecci
    python scripts/test_delta_lake.py --all
"""
import argparse
import sys
import os

try:
    import deltalake
    import pandas as pd
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install deltalake pandas")
    sys.exit(1)


# MinIO configuration - adjust host for local vs k8s testing
MINIO_HOST = os.environ.get("MINIO_HOST", "localhost")
MINIO_PORT = os.environ.get("MINIO_PORT", "9000")
MINIO_ENDPOINT = f"http://{MINIO_HOST}:{MINIO_PORT}"
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")

# Delta Lake storage options
STORAGE_OPTIONS = {
    "AWS_ENDPOINT_URL": MINIO_ENDPOINT,
    "AWS_ACCESS_KEY_ID": MINIO_ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": MINIO_SECRET_KEY,
    "AWS_REGION": "us-east-1",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    "AWS_ALLOW_HTTP": "true",
}

# Delta Lake table paths
DELTA_TABLES = {
    "tfd": {
        "name": "Transaction Fraud Detection",
        "path": "s3://lakehouse/delta/transaction_fraud_detection",
    },
    "eta": {
        "name": "Estimated Time of Arrival",
        "path": "s3://lakehouse/delta/estimated_time_of_arrival",
    },
    "ecci": {
        "name": "E-Commerce Customer Interactions",
        "path": "s3://lakehouse/delta/e_commerce_customer_interactions",
    },
}


def test_delta_table(table_key: str) -> bool:
    """Test a single Delta Lake table."""
    table_info = DELTA_TABLES.get(table_key)
    if not table_info:
        print(f"ERROR: Unknown table key: {table_key}")
        return False

    name = table_info["name"]
    path = table_info["path"]

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Path: {path}")
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"{'='*60}")

    try:
        # Try to open Delta table
        dt = deltalake.DeltaTable(path, storage_options=STORAGE_OPTIONS)

        # Get table metadata
        print(f"\n✓ Delta table exists!")
        print(f"  Version: {dt.version()}")
        print(f"  Files: {len(dt.files())}")

        # Get schema
        schema = dt.schema()
        print(f"  Columns: {len(schema.fields)}")
        print(f"  Schema:")
        for field in schema.fields:
            print(f"    - {field.name}: {field.type}")

        # Read data to pandas
        df = dt.to_pandas()
        print(f"\n✓ Data loaded successfully!")
        print(f"  Total rows: {len(df)}")

        if len(df) > 0:
            print(f"\n  Sample data (first 3 rows):")
            print(df.head(3).to_string(index=False))

            # Show column statistics
            print(f"\n  Column types:")
            for col, dtype in df.dtypes.items():
                print(f"    - {col}: {dtype}")
        else:
            print("  ⚠ Table is empty (no data written yet)")

        return True

    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Failed to read Delta table")

        if "NotFound" in error_msg or "NoSuchKey" in error_msg:
            print(f"  Reason: Table does not exist yet")
            print(f"  Hint: Spark Streaming may not have written any data yet.")
            print(f"        Wait for data generators to produce messages.")
        elif "Connection" in error_msg or "timeout" in error_msg:
            print(f"  Reason: Cannot connect to MinIO")
            print(f"  Hint: Check if MinIO is running and port-forwarded.")
            print(f"        Run: kubectl port-forward svc/coelho-realtime-minio 9000:9000 -n coelho-realtime")
        else:
            print(f"  Error: {error_msg}")

        return False


def main():
    parser = argparse.ArgumentParser(description="Test Delta Lake tables on MinIO")
    parser.add_argument(
        "--table", "-t",
        choices=["tfd", "eta", "ecci"],
        help="Test specific table (tfd, eta, ecci)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Test all tables"
    )
    args = parser.parse_args()

    print("Delta Lake Table Test")
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"Access Key: {MINIO_ACCESS_KEY[:4]}***")

    if args.table:
        success = test_delta_table(args.table)
        sys.exit(0 if success else 1)
    elif args.all:
        results = {}
        for key in DELTA_TABLES:
            results[key] = test_delta_table(key)

        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        for key, success in results.items():
            status = "✓ OK" if success else "✗ FAILED"
            print(f"  {DELTA_TABLES[key]['name']}: {status}")

        all_success = all(results.values())
        sys.exit(0 if all_success else 1)
    else:
        # Default: test TFD
        print("\nNo table specified, testing Transaction Fraud Detection...")
        print("Use --all to test all tables, or --table <tfd|eta|ecci>")
        success = test_delta_table("tfd")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
