import json
import time
import random
import uuid
from kafka import KafkaProducer
from faker import Faker
import datetime
import click
from pprint import pprint
import os


KAFKA_HOST = os.environ["KAFKA_HOST"]


fake = Faker()

KAFKA_TOPIC = 'transaction_fraud_detection'
KAFKA_BROKERS = f'{KAFKA_HOST}:9092'

# --- Constants ---
# Realistic choices for categorical features
TRANSACTION_TYPES = ['purchase', 'withdrawal', 'transfer', 'payment', 'deposit']
PAYMENT_METHODS = ['credit_card', 'debit_card', 'paypal', 'bank_transfer', 'crypto']
CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'BRL'] # Added BRL based on location context
PRODUCT_CATEGORIES = [
    'electronics', 'clothing', 'groceries', 'travel', 'services',
    'digital_goods', 'luxury_items', 'gambling', 'other'
]
DEVICE_OS = ['iOS', 'Android', 'Windows', 'macOS', 'Linux', 'Other']
BROWSERS = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Opera', 'Other']

def create_producer():
    # Retry connection logic with metadata readiness check
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers = KAFKA_BROKERS,
                value_serializer = lambda v: json.dumps(v).encode('utf-8'),
                client_id = f"{KAFKA_TOPIC}_client",
                request_timeout_ms = 30000,
                metadata_max_age_ms = 10000,
                reconnect_backoff_ms = 1000,
                reconnect_backoff_max_ms = 10000
            )
            # Wait for metadata to ensure Kafka is fully ready
            print("Checking Kafka metadata availability...")
            producer.partitions_for(KAFKA_TOPIC)  # This will block until metadata is available or timeout
            print("Kafka Producer connected and ready!")
            return producer
        except Exception as e:
            retry_count += 1
            print(f"Error connecting to Kafka (attempt {retry_count}/{max_retries}): {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
    raise Exception(f"Failed to connect to Kafka after {max_retries} attempts")


def generate_transaction(
        fraud_probability,
        account_age_days_limit,
        merchant_id_limit,
        cvv_provided_missing_probability,
        billing_address_match_missing_probability,
        high_value_fraud_probability):
    """
    Generates a simulated financial transaction with realistic features
    for fraud detection using incremental learning.
    """
    user_id = fake.uuid4() # Keep user_id consistent for potential future stateful features
    is_fraud = False # Default to not fraud
    base_random = random.random()
    # --- Simulate User Profile ---
    # Simulate account age - newer accounts might be slightly riskier
    account_age_days = random.randint(0, 365 * account_age_days_limit) # Account age from 0 days to limit years
    # --- Simulate Basic Transaction Info ---
    transaction_type = random.choice(TRANSACTION_TYPES)
    payment_method = random.choice(PAYMENT_METHODS)
    merchant_id = f'merchant_{random.randint(1, merchant_id_limit)}' # Increased merchant variety
    currency = random.choice(CURRENCIES)
    product_category = random.choice(PRODUCT_CATEGORIES)
    # --- Simulate Device and Context ---
    device_info = {
        'os': random.choice(DEVICE_OS),
        'browser': random.choice(BROWSERS)
    }
    user_agent = fake.user_agent()
    ip_address = fake.ipv4()
    location = {
        'lat': float(fake.latitude()),
        'lon': float(fake.longitude())
    }
    # --- Simulate Flags often used in Fraud Detection ---
    # Simulate CVV presence (might be missing in fraudulent attempts)
    cvv_provided = random.choices(
        [True, False], 
        weights = [1 - cvv_provided_missing_probability, cvv_provided_missing_probability], 
        k = 1)[0]
    # Simulate billing address match (mismatch can be a red flag)
    billing_address_match = random.choices(
        [True, False], 
        weights = [1 - billing_address_match_missing_probability, billing_address_match_missing_probability], 
        k = 1)[0]
    # --- Determine Fraud Status & Adjust Features ---
    # Initial determination based on probability
    if base_random < fraud_probability:
        is_fraud = True
    # --- Generate Amount (influenced by fraud status) ---
    if is_fraud:
        # Fraudulent transactions might be very high, or sometimes suspiciously small (testing cards)
        if random.random() < high_value_fraud_probability: # 80% chance of high-value fraud
             amount = round(random.uniform(300.0, 6000.0), 2)
        else: # 20% chance of low-value fraud (e.g., card testing)
             amount = round(random.uniform(1.0, 50.0), 2)
        # Increase probability of suspicious flags for fraudulent transactions
        if random.random() < 0.4: # 40% chance CVV is missing in fraud
            cvv_provided = False
        if random.random() < 0.3: # 30% chance billing address mismatches in fraud
            billing_address_match = False
        # Fraud might target specific categories more often
        if random.random() < 0.2:
             product_category = random.choice([
                'electronics', 
                'luxury_items', 
                'digital_goods', 
                'gambling'])
        # Fraud might originate from newer accounts more often
        if random.random() < 0.15:
            account_age_days = random.randint(0, 30) # Fraud from accounts < 1 month old
        # Fraud might use certain payment methods more
        if random.random() < 0.1:
            payment_method = random.choice(['crypto', 'credit_card']) # Example focus
    else:
        # Normal transaction amounts
        amount = round(random.uniform(5.0, 500.0), 2)
        # Ensure flags are more likely normal for non-fraud
        if not cvv_provided and random.random() < 0.9: # High chance CVV is present if not fraud
             cvv_provided = True
        if not billing_address_match and random.random() < 0.8: # High chance billing matches if not fraud
             billing_address_match = True
    # --- Assemble the Transaction Record ---
    transaction = {
        'transaction_id': str(uuid.uuid4()),
        'user_id': user_id,
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(), # ISO format UTC timestamp
        'amount': amount,
        'currency': currency,
        'merchant_id': merchant_id,
        'product_category': product_category,
        'transaction_type': transaction_type,
        'payment_method': payment_method,
        'location': location,
        'ip_address': ip_address,
        'device_info': device_info, # Nested structure for device details
        'user_agent': user_agent,
        'account_age_days': account_age_days,
        'cvv_provided': cvv_provided, # Boolean flag
        'billing_address_match': billing_address_match, # Boolean flag
        # CRUCIAL: Include the ground truth label for training the River model
        'is_fraud': int(is_fraud) # Use 1 for fraud, 0 for non-fraud
    }
    return transaction


@click.command()
@click.option(
    '--fraud-probability', 
    default = 0.01, 
    type = float, 
    help = 'The probability of a transaction being fraudulent.')
@click.option(
    '--account-age-days-limit', 
    default = 5, 
    type = int, 
    help = 'The maximum age of an account in years.')
@click.option(
    '--merchant-id-limit', 
    default = 200, 
    type = int, 
    help = 'The maximum number of unique merchant IDs to use.')
@click.option(
    '--cvv-provided-missing-probability',
    default = 0.05,
    type = float,
    help = 'The probability of a CVV being missing.')
@click.option(
    '--billing-address-match-missing-probability',
    default = 0.1,
    type = float,
    help = 'The probability of a billing address mismatch.')
@click.option(
    '--high-value-fraud-probability',
    default = 0.8,
    type = float,
    help = 'The probability of a high-value transaction being fraudulent.'
)
def run_producer(
    fraud_probability,
    account_age_days_limit,
    merchant_id_limit,
    cvv_provided_missing_probability,
    billing_address_match_missing_probability,
    high_value_fraud_probability):
    producer = create_producer()
    print("Starting to send transaction events...")
    last_print_time = time.time()
    message_count = 0
    try:
        while True:
            transaction = generate_transaction(
                fraud_probability,
                account_age_days_limit,
                merchant_id_limit,
                cvv_provided_missing_probability,
                billing_address_match_missing_probability,
                high_value_fraud_probability
            )
            producer.send(KAFKA_TOPIC, value = transaction)
            message_count += 1
            # Print sample every 60 seconds
            current_time = time.time()
            if current_time - last_print_time >= 60:
                print(f"\n###--- Transaction Fraud Detection ({message_count} msgs sent) ---###")
                pprint(transaction)
                last_print_time = current_time
            # Simulate varying transaction rate
            time.sleep(random.uniform(0.05, 0.5))
    except KeyboardInterrupt:
        print("Stopping producer.")
    finally:
        if producer:
            producer.flush() # Ensure all messages are sent
            producer.close()
            print("Producer closed.")

if __name__ == "__main__":
    run_producer()