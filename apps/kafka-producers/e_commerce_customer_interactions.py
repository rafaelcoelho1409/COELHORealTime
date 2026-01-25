import orjson
import time
import random
import uuid
from kafka import KafkaProducer
from faker import Faker
import datetime
import click
import os
from pprint import pprint


KAFKA_HOST = os.environ["KAFKA_HOST"]
KAFKA_TOPIC = 'e_commerce_customer_interactions' # Topic name remains the same
KAFKA_BROKERS = f'{KAFKA_HOST}:9092'

# Use US English locale for fake data
fake = Faker('en_US')

# --- Constants ---
EVENT_TYPES = ['page_view', 'add_to_cart', 'purchase', 'search', 'leave_review']
# Relevant product categories for US E-commerce
PRODUCT_CATEGORIES = [
    'Electronics', 'Fashion & Apparel', 'Home & Garden', 'Beauty & Personal Care',
    'Sports & Outdoors', 'Books', 'Grocery & Gourmet Food', 'Automotive',
    'Toys & Games', 'Computers', 'Pet Supplies', 'Health & Household'
]
DEVICE_TYPES = ['Mobile', 'Desktop', 'Tablet']
BROWSERS = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Opera', 'Other']
OS_TYPES = ['Android', 'iOS', 'Windows', 'macOS', 'Linux', 'Other']

# Simulate coordinates roughly within the Houston metropolitan area
LAT_BOUNDS_HOU = (29.5, 30.1)  # Approximate North/South bounds for Houston metro
LON_BOUNDS_HOU = (-95.8, -95.0) # Approximate West/East bounds for Houston metro

# Limited set of referrer URLs to avoid dropdown performance issues
REFERRER_URLS = [
    'direct',
    'google.com',
    'facebook.com',
    'amazon.com',
    'instagram.com',
    'twitter.com',
    'youtube.com',
    'tiktok.com',
    'pinterest.com',
    'reddit.com',
    'linkedin.com',
    'bing.com',
    'yahoo.com',
    'email_campaign',
    'affiliate_link',
]

# Keep track of active sessions to make event sequences more realistic
active_sessions = {} # session_id -> {customer_id, last_event_time, events_in_session, current_product_focus}
MAX_ACTIVE_SESSIONS = 500 # Limit memory usage
SESSION_TIMEOUT_SECONDS = 15 * 60 # 15 minutes

def create_producer():
    """Creates the Kafka producer with retry logic and metadata readiness check."""
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers = KAFKA_BROKERS,
                value_serializer = lambda v: orjson.dumps(v),
                client_id = f"{KAFKA_TOPIC}_client",
                request_timeout_ms = 30000,
                metadata_max_age_ms = 10000,
                reconnect_backoff_ms = 1000,
                reconnect_backoff_max_ms = 10000
            )
            # Wait for metadata to ensure Kafka is fully ready
            print("Checking Kafka metadata availability...")
            producer.partitions_for(KAFKA_TOPIC)
            print("Kafka Producer connected and ready!")
            return producer
        except Exception as e:
            retry_count += 1
            print(f"Error connecting to Kafka (attempt {retry_count}/{max_retries}): {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
    raise Exception(f"Failed to connect to Kafka after {max_retries} attempts")

def generate_customer_event(event_rate_per_minute):
    """
    Generates a simulated customer interaction event for an e-commerce platform.
    Tries to maintain some session consistency. Focus on Houston, TX region and English data.
    """
    global active_sessions
    now = datetime.datetime.now(datetime.timezone.utc)
    now_ts = now.timestamp()
    # Clean up timed-out sessions infrequently
    if random.random() < 0.01:
        active_sessions = {
            sid: data for sid, data in active_sessions.items()
            if (now_ts - data['last_event_time']) < SESSION_TIMEOUT_SECONDS
        }
    session_id = None
    customer_id = None
    session_data = None
    # Decide whether to continue an existing session or start a new one
    if active_sessions and random.random() < 0.7: # 70% chance to continue existing session
        session_id = random.choice(list(active_sessions.keys()))
        session_data = active_sessions[session_id]
        customer_id = session_data['customer_id']
        # Check for timeout just in case cleanup didn't run
        if (now_ts - session_data['last_event_time']) >= SESSION_TIMEOUT_SECONDS:
            if session_id in active_sessions: # Check if key still exists before deleting
                 del active_sessions[session_id]
            session_id = None # Force new session
    # Start a new session if needed or if max sessions not reached
    if session_id is None and len(active_sessions) < MAX_ACTIVE_SESSIONS:
        session_id = str(uuid.uuid4())
        customer_id = fake.uuid4() # New customer for new session (simplification)
        session_data = {
            'customer_id': customer_id,
            'last_event_time': now_ts,
            'events_in_session': 0,
            'current_product_focus': None # Track product viewed/added
        }
        active_sessions[session_id] = session_data
    # If we couldn't get a session (e.g., max reached and couldn't reuse), skip event
    if session_id is None:
        return None
    # --- Determine Event Type based on session state ---
    event_type = None
    events_so_far = session_data['events_in_session']
    product_focus = session_data['current_product_focus']
    # Simple state machine for event flow
    if events_so_far == 0:
        event_type = random.choices(
            ['page_view', 'search'], 
            weights = [0.8, 0.2], 
            k = 1)[0]
    elif product_focus and random.random() < 0.4: # 40% chance to interact with focused product
        event_type = random.choices(
            ['add_to_cart', 'page_view', 'leave_review'], # Can view related items or add
            weights = [0.6, 0.35, 0.05], 
            k = 1)[0]
        if event_type == 'add_to_cart' and random.random() < 0.3: # 30% chance to purchase after adding
             event_type = 'purchase'
    else: # Generic next action
        event_type = random.choices(
            ['page_view', 'search', 'add_to_cart'],
            weights = [0.6, 0.2, 0.2], 
            k = 1)[0]
    # --- Generate Event Details ---
    event_id = str(uuid.uuid4())
    timestamp = now.isoformat()
    product_id = None
    product_category = None
    price = None
    quantity = None
    page_url = None
    referrer_url = None
    time_on_page_seconds = None
    search_query = None
    # Generate details based on event type
    if event_type in ['page_view', 'add_to_cart', 'purchase', 'leave_review']:
        product_id = f'prod_{random.randint(1000, 1100)}' # Limited to 100 products for dropdown performance
        product_category = random.choice(PRODUCT_CATEGORIES)
        # Adjust price ranges slightly for USD context if desired, keeping it broad
        price = round(random.uniform(5.0, 2500.0), 2)
        # Use generic .com domain
        page_url = f"https://example.com/{product_category.lower().replace(' & ', '-').replace(' ', '-')}/{product_id}"
        time_on_page_seconds = random.randint(5, 300) # 5 seconds to 5 minutes
        # Update session focus
        session_data['current_product_focus'] = {
            'id': product_id, 'category': product_category, 'price': price
            }
        if event_type in ['add_to_cart', 'purchase']:
            quantity = random.randint(1, 4) # Slightly lower max quantity perhaps
        if event_type == 'purchase':
            # Simulate checkout page view before purchase completion
            time_on_page_seconds = random.randint(30, 180)
            # Clear focus after purchase
            session_data['current_product_focus'] = None
        if event_type == 'leave_review':
            # Assume review happens after purchase, clear focus
            session_data['current_product_focus'] = None
    elif event_type == 'search':
        # Generate more typical English search terms
        search_query = ' '.join(fake.words(nb = random.randint(1, 4)))
        page_url = f"https://example.com/search?q={search_query.replace(' ', '+')}"
        time_on_page_seconds = random.randint(5, 60)
        # Clear focus on search
        session_data['current_product_focus'] = None
    # Common fields
    device_type = random.choice(DEVICE_TYPES)
    browser = random.choice(BROWSERS)
    os_type = random.choice(OS_TYPES)
    # Use Houston coordinates
    location = {
        'lat': round(random.uniform(LAT_BOUNDS_HOU[0], LAT_BOUNDS_HOU[1]), 3),
        'lon': round(random.uniform(LON_BOUNDS_HOU[0], LON_BOUNDS_HOU[1]), 3)
    }
    # Simulate referrer - using fixed list for dropdown performance
    if events_so_far == 0:
        referrer_url = random.choice(REFERRER_URLS)
    # Update session state
    session_data['last_event_time'] = now_ts
    session_data['events_in_session'] += 1
    # --- Assemble the Event Record ---
    interaction_event = {
        'event_id': event_id,
        'customer_id': customer_id,
        'session_id': session_id,
        'timestamp': timestamp,
        'event_type': event_type,
        'product_id': product_id,
        'product_category': product_category, # Now in English
        'price': price, # USD context (range adjusted slightly)
        'quantity': quantity,
        'page_url': page_url, # .com domain
        'referrer_url': referrer_url,
        'device_info': { # Nested structure
             'device_type': device_type,
             'browser': browser,
             'os': os_type
        },
        'location': location, # Houston coordinates
        'time_on_page_seconds': time_on_page_seconds,
        'search_query': search_query, # English terms
        # Include session sequence number for potential analysis
        'session_event_sequence': session_data['events_in_session']
    }
    return interaction_event

@click.command()
@click.option(
    '--event-rate-per-minute',
    default = 600, # Target 10 events per second on average
    type = int,
    help = 'Target number of events to generate per minute.')
def run_producer(event_rate_per_minute):
    """Runs the Kafka producer to continuously generate customer interaction events."""
    producer = create_producer()
    print(f"Starting to send customer interaction events (target rate: {event_rate_per_minute}/min)...")
    # Calculate sleep time based on target rate
    target_events_per_second = event_rate_per_minute / 60.0
    base_sleep_time = 1.0 / target_events_per_second if target_events_per_second > 0 else 1.0
    last_print_time = time.time()
    message_count = 0
    try:
        while True:
            # Generate event
            interaction_event = generate_customer_event(event_rate_per_minute)
            if interaction_event:
                producer.send(KAFKA_TOPIC, value = interaction_event)
                message_count += 1
                # Print sample every 60 seconds
                current_time = time.time()
                if current_time - last_print_time >= 60:
                    print(f"\n###--- E-Commerce Customer Interactions ({message_count} msgs sent) ---###")
                    pprint(interaction_event)
                    last_print_time = current_time
            # Adjust sleep time slightly to simulate variability and approximate target rate
            sleep_time = max(0.001, random.gauss(base_sleep_time, base_sleep_time * 0.3))
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopping producer.")
    finally:
        if producer:
            producer.flush() # Ensure all messages are sent
            producer.close()
            print("Producer closed.")

if __name__ == "__main__":
    run_producer()
