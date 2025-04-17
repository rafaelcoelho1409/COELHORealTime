import json
import time
import random
import uuid
from kafka import KafkaProducer
from faker import Faker
import datetime

fake = Faker()

KAFKA_TOPIC = 'transactions'
KAFKA_BROKERS = 'kafka-producer:9092' # Use the internal Docker DNS name

# --- Constants ---
FRAUD_PROBABILITY = 0.01 # 1% base probability of a transaction being fraudulent
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
    # Retry connection logic
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers = KAFKA_BROKERS,
                value_serializer = lambda v: json.dumps(v).encode('utf-8'),
                client_id = 'fraud-producer'
            )
            print("Kafka Producer connected!")
            return producer
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

def generate_transaction():
    """
    Generates a simulated financial transaction with realistic features
    for fraud detection using incremental learning.
    """
    user_id = fake.uuid4() # Keep user_id consistent for potential future stateful features
    is_fraud = False # Default to not fraud
    base_random = random.random()
    # --- Simulate User Profile ---
    # Simulate account age - newer accounts might be slightly riskier
    account_age_days = random.randint(0, 365 * 5) # Account age from 0 days to 5 years
    # --- Simulate Basic Transaction Info ---
    transaction_type = random.choice(TRANSACTION_TYPES)
    payment_method = random.choice(PAYMENT_METHODS)
    merchant_id = f'merchant_{random.randint(1, 200)}' # Increased merchant variety
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
    cvv_provided = random.choices([True, False], weights=[0.95, 0.05], k=1)[0]
    # Simulate billing address match (mismatch can be a red flag)
    billing_address_match = random.choices([True, False], weights=[0.9, 0.1], k=1)[0]
    # --- Determine Fraud Status & Adjust Features ---
    # Initial determination based on probability
    if base_random < FRAUD_PROBABILITY:
        is_fraud = True
    # --- Generate Amount (influenced by fraud status) ---
    if is_fraud:
        # Fraudulent transactions might be very high, or sometimes suspiciously small (testing cards)
        if random.random() < 0.8: # 80% chance of high-value fraud
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
             product_category = random.choice(['electronics', 'luxury_items', 'digital_goods', 'gambling'])
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

if __name__ == "__main__":
    producer = create_producer()
    print("Starting to send transaction events...")
    try:
        while True:
            transaction = generate_transaction()
            producer.send(KAFKA_TOPIC, value = transaction)
            # Limit console output frequency for readability
            if random.random() < 0.1:
                 print(f"Sent: {transaction}")
            # Simulate varying transaction rate
            time.sleep(random.uniform(0.05, 0.5))
    except KeyboardInterrupt:
        print("Stopping producer.")
    finally:
        if producer:
            producer.flush() # Ensure all messages are sent
            producer.close()
            print("Producer closed.")