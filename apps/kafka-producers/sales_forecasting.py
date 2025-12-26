import json
import time
import random
import uuid
from kafka import KafkaProducer
from faker import Faker
import datetime
import click
from pprint import pprint
import math
import os


KAFKA_HOST = os.environ["KAFKA_HOST"]


fake = Faker()

# --- Configuration ---
KAFKA_TOPIC = 'sales_forecasting'
KAFKA_BROKERS = f'{KAFKA_HOST}:9092'

# --- Constants for Data Generation ---
PRODUCT_IDS = [f'SKU_{str(i).zfill(5)}' for i in range(1, 21)] # 20 unique products
STORE_IDS = [f'STORE_{str(i).zfill(3)}' for i in range(1, 6)] # 5 unique stores
INITIAL_DATE = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=365 * 2) # Start 2 years ago to show long term trends

# Base characteristics for products (can be made more complex)
PRODUCT_CATALOG = {
    pid: {
        'base_price': round(random.uniform(5.0, 200.0), 2),
        'base_demand': random.randint(5, 50), # Average units sold per typical interval without promotions
        'price_elasticity': random.uniform(0.5, 2.0), # How much demand changes with price
        'seasonality_profile': random.choice(['strong_winter', 'strong_summer', 'weekend_peak', 'steady']),
        'trend_factor': random.uniform(-0.05, 0.1) / 365, # Small daily trend factor (positive or negative)
        'promotion_uplift_factor': random.uniform(1.2, 3.0) # How much promotions boost sales
    }
    for pid in PRODUCT_IDS
}

# --- Helper Functions ---
def get_seasonal_multiplier(current_time, profile_name):
    month = current_time.month
    weekday = current_time.weekday() # Monday is 0 and Sunday is 6

    if profile_name == 'strong_winter':
        if month in [11, 12, 1, 2]: return random.uniform(1.5, 2.5)
        elif month in [6, 7, 8]: return random.uniform(0.5, 0.8)
        return random.uniform(0.9, 1.1)
    elif profile_name == 'strong_summer':
        if month in [6, 7, 8]: return random.uniform(1.5, 2.5)
        elif month in [11, 12, 1, 2]: return random.uniform(0.5, 0.8)
        return random.uniform(0.9, 1.1)
    elif profile_name == 'weekend_peak':
        if weekday >= 5: return random.uniform(1.3, 2.0) # Fri, Sat, Sun
        return random.uniform(0.7, 1.0)
    else: # steady
        return random.uniform(0.95, 1.05)

def create_producer():
    """Creates the Kafka producer with retry logic and metadata readiness check."""
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                client_id=f"{KAFKA_TOPIC}_client",
                request_timeout_ms=30000,
                metadata_max_age_ms=10000,
                reconnect_backoff_ms=1000,
                reconnect_backoff_max_ms=10000
            )
            # Wait for metadata to ensure Kafka is fully ready
            print(f"Checking Kafka metadata availability for {KAFKA_TOPIC}...")
            producer.partitions_for(KAFKA_TOPIC)
            print(f"Kafka Producer for {KAFKA_TOPIC} connected and ready!")
            return producer
        except Exception as e:
            retry_count += 1
            print(f"Error connecting to Kafka for {KAFKA_TOPIC} (attempt {retry_count}/{max_retries}): {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
    raise Exception(f"Failed to connect to Kafka after {max_retries} attempts")

def generate_sales_event(current_sim_time, days_elapsed, concept_drift_stage):
    """
    Generates a simulated sales event with features relevant for demand forecasting.
    `concept_drift_stage` can be used to alter generation logic over time.
    """
    product_id = random.choice(PRODUCT_IDS)
    store_id = random.choice(STORE_IDS)
    product_info = PRODUCT_CATALOG[product_id]

    # --- Base Demand & Trend ---
    # Apply long-term trend
    trend_multiplier = 1 + (product_info['trend_factor'] * days_elapsed)
    # Introduce a concept drift: e.g., after 1.5 years, some products' trends might reverse or accelerate
    if concept_drift_stage > 1 and product_id in PRODUCT_IDS[:5]: # First 5 products affected by major drift
        trend_multiplier *= 1.5 # Accelerated trend
    elif concept_drift_stage > 1 and product_id in PRODUCT_IDS[5:10]:
        trend_multiplier *= 0.5 # Dampened trend or reversal if negative

    current_base_demand = product_info['base_demand'] * trend_multiplier
    current_base_demand = max(1, current_base_demand) # Ensure demand is at least 1

    # --- Seasonality ---
    seasonal_multiplier = get_seasonal_multiplier(current_sim_time, product_info['seasonality_profile'])
    # Concept drift: Seasonality profile might change for some products
    if concept_drift_stage > 2 and product_id in PRODUCT_IDS[10:15]:
        seasonal_multiplier = get_seasonal_multiplier(current_sim_time, 'weekend_peak') # Switch profile

    # --- Promotions (Exogenous Variable) ---
    is_promotion_active = False
    promotion_id = None
    price_multiplier = 1.0
    promotion_uplift = 1.0

    # Simulate promotions (e.g., 10% chance a product is on promotion)
    # Promotions might become more/less effective (concept drift)
    promotion_effectiveness_factor = 1.0
    if concept_drift_stage > 0: # After some time, promotions change
        promotion_effectiveness_factor = random.uniform(0.8, 1.2)


    if random.random() < 0.10: # 10% chance of active promotion for this product at this time
        is_promotion_active = True
        promotion_id = f"PROMO_{current_sim_time.year}_{current_sim_time.month}_{product_id[:3]}"
        discount = random.uniform(0.05, 0.30) # 5% to 30% discount
        price_multiplier = 1.0 - discount
        # Uplift due to promotion, affected by elasticity and base uplift factor and drift
        promotion_uplift = product_info['promotion_uplift_factor'] * (1 + (discount * product_info['price_elasticity']))
        promotion_uplift *= promotion_effectiveness_factor
        promotion_uplift = max(1.0, promotion_uplift) # Ensure it doesn't reduce sales


    # --- Calculate Quantity Sold ---
    quantity_sold = current_base_demand * seasonal_multiplier * promotion_uplift
    # Add some random noise
    quantity_sold *= random.uniform(0.85, 1.15)
    quantity_sold = max(0, int(round(quantity_sold))) # Ensure non-negative integer

    # --- Calculate Financials ---
    unit_price = round(product_info['base_price'] * price_multiplier, 2)
    total_sales_amount = round(quantity_sold * unit_price, 2)

    # --- Other Features (Exogenous Variables for SNARIMAX) ---
    day_of_week = current_sim_time.weekday() # Monday=0, Sunday=6
    month = current_sim_time.month
    is_holiday = False # Basic holiday simulation
    if (month == 12 and current_sim_time.day >= 15 and current_sim_time.day <= 25) or \
       (month == 1 and current_sim_time.day == 1) or \
       (month == 7 and current_sim_time.day >= 1 and current_sim_time.day <= 7): # Example holiday periods
        is_holiday = True
        if not is_promotion_active: # Holidays often boost sales even without specific promotions
            quantity_sold = int(quantity_sold * random.uniform(1.1, 1.5))
            total_sales_amount = round(quantity_sold * unit_price, 2)


    # --- Assemble the Event Record ---
    event = {
        'event_id': str(uuid.uuid4()),
        'timestamp': current_sim_time.isoformat(),
        'product_id': product_id,
        'store_id': store_id,
        'quantity_sold': quantity_sold, # This is a key target for forecasting
        'unit_price': unit_price,
        'total_sales_amount': total_sales_amount, # Financial impact
        'is_promotion_active': is_promotion_active,
        'promotion_id': promotion_id,
        'day_of_week': day_of_week, # 0-6
        'month': month, # 1-12
        'is_holiday': is_holiday,
        'concept_drift_stage': concept_drift_stage, # To observe how drift impacts data
        # Potentially add other exogenous variables like:
        # 'competitor_price_index': round(random.uniform(0.9, 1.1), 2),
        # 'marketing_spend_today': round(random.uniform(0, 1000), 2) if is_promotion_active else 0,
        # 'weather_avg_temp': round(random.uniform(-5, 35), 1) # if relevant for products
    }
    return event

@click.command()
@click.option(
    '--time-scale-factor',
    default=3600, # 1 hour in seconds. So 1 sec real time = 1 hour sim time.
    type=int,
    help='How many simulated seconds pass for each real second of generation.'
)
@click.option(
    '--max-events',
    default=0, # 0 means infinite
    type=int,
    help='Maximum number of events to generate. 0 for continuous.'
)
@click.option(
    '--initial-delay-s',
    default=0.1,
    type=float,
    help='Initial delay between messages in seconds (gets slightly randomized).'
)
def run_producer(time_scale_factor, max_events, initial_delay_s):
    producer = create_producer()
    print("Starting to send sales events for demand forecasting...")

    current_sim_time = INITIAL_DATE
    events_sent = 0
    drift_change_interval_days = 180 # Change concept drift stage every ~6 months
    last_print_time = time.time()

    try:
        while True:
            if max_events > 0 and events_sent >= max_events:
                print(f"Generated maximum of {max_events} events. Stopping.")
                break

            days_elapsed = (current_sim_time - INITIAL_DATE).total_seconds() / (24 * 3600)
            concept_drift_stage = int(days_elapsed / drift_change_interval_days)

            event = generate_sales_event(current_sim_time, days_elapsed, concept_drift_stage)
            producer.send(KAFKA_TOPIC, value=event)
            events_sent += 1

            # Print sample every 60 seconds
            current_time = time.time()
            if current_time - last_print_time >= 60:
                print(f"\n###--- Sales Forecasting ({events_sent} msgs sent) ---###")
                pprint(event)
                last_print_time = current_time

            # --- Time Progression ---
            # Simulate processing for a store; multiple transactions can happen closely
            # This means multiple events might share nearly the same timestamp (within a minute or hour)
            # before advancing the main sim clock significantly.
            if random.random() < 0.7: # 70% chance next event is within the same hour
                time_increment_seconds = random.randint(1, 60 * 5) # 1 to 5 minutes
            else: # 30% chance we jump to next hour or a bit more
                time_increment_seconds = random.randint(60 * 30, 60 * 120) # 0.5 to 2 hours

            current_sim_time += datetime.timedelta(seconds=time_increment_seconds)

            # Control real-time send rate
            sleep_duration = time_increment_seconds / time_scale_factor
            sleep_duration = max(0.001, sleep_duration * random.uniform(0.8, 1.2)) # Add jitter
            # If initial_delay_s is set, use it for the first few events or if sleep_duration is too small
            actual_sleep = max(sleep_duration, initial_delay_s if events_sent < 100 else 0.001)
            time.sleep(actual_sleep)


    except KeyboardInterrupt:
        print("Stopping producer.")
    finally:
        if producer:
            print("Flushing messages...")
            producer.flush()
            producer.close()
            print("Producer closed.")

if __name__ == "__main__":
    run_producer()