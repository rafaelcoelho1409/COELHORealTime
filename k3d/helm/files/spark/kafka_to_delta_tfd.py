"""
Kafka to Delta Lake Streaming Job - Transaction Fraud Detection

Reads from Kafka topic, writes to Delta Lake on MinIO.
Supports graceful shutdown and checkpointing for exactly-once semantics.
"""
import os
import signal
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, BooleanType
)

# Configuration from environment
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "coelho-realtime-kafka:9092")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://coelho-realtime-minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
DELTA_PATH = os.getenv("DELTA_PATH", "s3a://lakehouse/delta/transaction_fraud_detection")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "s3a://lakehouse/checkpoints/tfd")
KAFKA_TOPIC = "transaction_fraud_detection"

# Graceful shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Define schema for Transaction Fraud Detection events
# Matches exactly what Kafka producer sends
tfd_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("merchant_id", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("payment_method", StringType(), True),
    StructField("location", StringType(), True),  # JSON dict stored as string
    StructField("ip_address", StringType(), True),
    StructField("device_info", StringType(), True),  # JSON dict stored as string
    StructField("user_agent", StringType(), True),
    StructField("account_age_days", IntegerType(), True),
    StructField("cvv_provided", BooleanType(), True),
    StructField("billing_address_match", BooleanType(), True),
    StructField("is_fraud", IntegerType(), True),
])


def main():
    print("Starting Kafka to Delta Lake streaming job...")
    print(f"Kafka Bootstrap: {KAFKA_BOOTSTRAP}")
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"Delta Path: {DELTA_PATH}")
    print(f"Checkpoint Path: {CHECKPOINT_PATH}")
    # Build Spark session with Delta Lake and S3 support
    builder = SparkSession.builder \
        .appName("KafkaToDelta-TransactionFraudDetection") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("Spark session created successfully")
    try:
        # Read from Kafka
        kafka_df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
            .option("subscribe", KAFKA_TOPIC) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .load()
        # Parse JSON and extract fields
        parsed_df = kafka_df \
            .selectExpr("CAST(value AS STRING) as json_value") \
            .select(from_json(col("json_value"), tfd_schema).alias("data")) \
            .select("data.*")
        # Write to Delta Lake
        query = parsed_df.writeStream \
            .format("delta") \
            .outputMode("append") \
            .option("checkpointLocation", CHECKPOINT_PATH) \
            .option("mergeSchema", "true") \
            .start(DELTA_PATH)
        print(f"Streaming query started, writing to {DELTA_PATH}")
        # Wait for termination or shutdown signal
        while not shutdown_requested:
            if query.awaitTermination(timeout = 5):
                break
        if shutdown_requested:
            print("Graceful shutdown initiated, stopping query...")
            query.stop()
            print("Query stopped successfully")
    except Exception as e:
        print(f"Error in streaming job: {e}")
        raise
    finally:
        spark.stop()
        print("Spark session stopped")


if __name__ == "__main__":
    main()
