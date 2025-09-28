import json
import time
import random
import uuid
from kafka import KafkaProducer
from faker import Faker
import datetime
import click
import math # Needed for Haversine distance
from pprint import pprint

fake = Faker('en_US') # Using Brazilian Portuguese locale for more relevant fake data

KAFKA_TOPIC = 'estimated_time_of_arrival' # Changed topic name
KAFKA_BROKERS = 'kafka:9092' # Assuming same Kafka setup

# --- Constants ---
VEHICLE_TYPES = ['Sedan', 'SUV', 'Hatchback', 'Motorcycle', 'Van']
WEATHER_CONDITIONS = ['Clear', 'Clouds', 'Rain', 'Heavy Rain', 'Fog', 'Thunderstorm']
# Simulate coordinates roughly within the Houston metropolitan area
LAT_BOUNDS = (29.5, 30.1)  # Approximate North/South bounds for Houston metro
LON_BOUNDS = (-95.8, -95.0) # Approximate West/East bounds for Houston metro (West longitude is negative)
AVG_SPEED_KMH = 40 # Average baseline speed in km/h for initial estimate

# --- Helper Function ---
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers.
    return c * r

def create_producer():
    """Creates the Kafka producer with retry logic."""
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers = KAFKA_BROKERS,
                value_serializer = lambda v: json.dumps(v).encode('utf-8'),
                client_id = f"{KAFKA_TOPIC}_client"
            )
            print("Kafka Producer connected!")
            return producer
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

def generate_eta_event(
        heavy_traffic_multiplier,
        bad_weather_probability,
        incident_probability):
    """
    Generates a simulated ride/delivery event for ETA prediction.
    Includes features known at the start and a simulated ground truth travel time.
    """
    trip_id = str(uuid.uuid4())
    driver_id = f'driver_{random.randint(1000, 5000)}'
    vehicle_id = f'vehicle_{random.randint(100, 999)}'
    vehicle_type = random.choice(VEHICLE_TYPES)
    # --- Simulate Origin and Destination ---
    origin_lat = round(random.uniform(LAT_BOUNDS[0], LAT_BOUNDS[1]), 6)
    origin_lon = round(random.uniform(LON_BOUNDS[0], LON_BOUNDS[1]), 6)
    dest_lat = round(random.uniform(LAT_BOUNDS[0], LAT_BOUNDS[1]), 6)
    dest_lon = round(random.uniform(LON_BOUNDS[0], LON_BOUNDS[1]), 6)
    # Ensure origin and destination are not identical (or extremely close)
    while abs(origin_lat - dest_lat) < 0.001 and abs(origin_lon - dest_lon) < 0.001:
        dest_lat = round(random.uniform(LAT_BOUNDS[0], LAT_BOUNDS[1]), 6)
        dest_lon = round(random.uniform(LON_BOUNDS[0], LON_BOUNDS[1]), 6)
    # --- Calculate Initial Estimates ---
    estimated_distance_km = round(haversine(origin_lat, origin_lon, dest_lat, dest_lon), 2)
    # Baseline travel time in seconds based on average speed
    # Add small noise to initial estimate to make it less perfect
    initial_estimated_travel_time_seconds = int((estimated_distance_km / AVG_SPEED_KMH) * 3600 * random.uniform(0.9, 1.1))
    # Ensure minimum travel time (e.g., 1 minute)
    initial_estimated_travel_time_seconds = max(60, initial_estimated_travel_time_seconds)
    # --- Simulate Factors Affecting Actual Travel Time ---
    current_time = datetime.datetime.now(datetime.timezone.utc)
    timestamp = current_time.isoformat()
    day_of_week = current_time.weekday() # Monday = 0, Sunday = 6
    hour_of_day = current_time.hour
    # Simulate Weather
    weather = random.choices(
        WEATHER_CONDITIONS,
        weights=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05], # Higher chance of Clear/Clouds
        k=1)[0]
    is_bad_weather = random.random() < bad_weather_probability and weather in ['Rain', 'Heavy Rain', 'Fog', 'Thunderstorm', 'Snow'] # Snow less likely in SP but kept for generality
    # Simulate Traffic Conditions (based on time)
    is_rush_hour = (6 <= hour_of_day <= 9) or (16 <= hour_of_day <= 19) # Typical rush hours
    traffic_factor = 1.0
    if is_rush_hour and day_of_week < 5: # Heavier traffic during weekday rush hours
        traffic_factor = random.uniform(1.3, heavy_traffic_multiplier)
    elif day_of_week >= 5: # Lighter traffic on weekends
        traffic_factor = random.uniform(0.8, 1.2)
    else: # Normal traffic during off-peak weekdays
        traffic_factor = random.uniform(0.9, 1.4)
    # Simulate Weather Impact
    weather_factor = 1.0
    if is_bad_weather:
        if weather == 'Rain': weather_factor = random.uniform(1.1, 1.4)
        elif weather == 'Heavy Rain': weather_factor = random.uniform(1.3, 1.8)
        elif weather == 'Fog': weather_factor = random.uniform(1.2, 1.6)
        elif weather == 'Thunderstorm': weather_factor = random.uniform(1.4, 2.0)
        # Add other conditions if needed
    # Simulate Random Incidents (accidents, unexpected delays)
    incident_delay_seconds = 0
    if random.random() < incident_probability:
        incident_delay_seconds = random.randint(300, 1800) # 5 to 30 minutes delay
    # Simulate Driver Factor (subtle influence)
    driver_rating = round(random.uniform(3.5, 5.0), 1) # Simulate ratings
    # Slightly adjust time based on rating (better drivers *might* be marginally faster/more efficient)
    driver_factor = 1.0 - (driver_rating - 4.5) * 0.05 # Max 2.5% faster for 5.0, 2.5% slower for 4.0
    # --- Calculate Simulated Actual Travel Time ---
    # Start with the *more accurate* baseline before noise was added
    base_time_seconds = (estimated_distance_km / AVG_SPEED_KMH) * 3600
    simulated_actual_travel_time_seconds = int(
        base_time_seconds * traffic_factor * weather_factor * driver_factor
        + incident_delay_seconds
        + random.uniform(-60, 60) # Add some final random noise (+/- 1 min)
    )
    # Ensure minimum travel time
    simulated_actual_travel_time_seconds = max(60, simulated_actual_travel_time_seconds)
    # --- Assemble the Event Record ---
    eta_event = {
        'trip_id': trip_id,
        'driver_id': driver_id,
        'vehicle_id': vehicle_id,
        'timestamp': timestamp, # Time the request/event was generated
        'origin': {'lat': origin_lat, 'lon': origin_lon},
        'destination': {'lat': dest_lat, 'lon': dest_lon},
        'estimated_distance_km': estimated_distance_km,
        'weather': weather,
        'temperature_celsius': round(random.uniform(15.0, 30.0), 1), # Example temp for SP
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'driver_rating': driver_rating,
        'vehicle_type': vehicle_type,
        # --- Features available for prediction at the start ---
        'initial_estimated_travel_time_seconds': initial_estimated_travel_time_seconds,
        # --- Ground Truth (Simulated) ---
        # This is what the incremental model will try to predict
        'simulated_actual_travel_time_seconds': simulated_actual_travel_time_seconds,
        # You could also include the simulated factors for analysis/debugging:
        'debug_traffic_factor': round(traffic_factor, 2),
        'debug_weather_factor': round(weather_factor, 2),
        'debug_incident_delay_seconds': incident_delay_seconds,
        'debug_driver_factor': round(driver_factor, 2)
    }
    return eta_event


@click.command()
@click.option(
    '--heavy-traffic-multiplier',
    default = 2.0,
    type = float,
    help = 'Maximum multiplier for travel time during heavy traffic rush hours (e.g., 2.0 means up to 2x longer).')
@click.option(
    '--bad-weather-probability',
    default = 0.1,
    type = float,
    help = 'Probability of encountering bad weather conditions (Rain, Fog, etc.).')
@click.option(
    '--incident-probability',
    default = 0.02,
    type = float,
    help = 'Probability of a random incident (e.g., accident) causing significant delay.')
def run_producer(
    heavy_traffic_multiplier, 
    bad_weather_probability, 
    incident_probability):
    """Runs the Kafka producer to continuously generate ETA prediction events."""
    producer = create_producer()
    print("Starting to send ETA prediction events...")
    try:
        while True:
            eta_event = generate_eta_event(
                heavy_traffic_multiplier,
                bad_weather_probability,
                incident_probability
            )
            producer.send(KAFKA_TOPIC, value = eta_event)
            # Limit console output frequency for readability
            if random.random() < 0.05:
                # Print only key info to keep console cleaner
                print("###--- Estimated Time of Arrival ---###")
                pprint(eta_event)
            # Simulate varying request rate (e.g., new ride request every 0.1 to 1 second)
            time.sleep(random.uniform(0.1, 1.0))
    except KeyboardInterrupt:
        print("Stopping producer.")
    finally:
        if producer:
            producer.flush() # Ensure all messages are sent
            producer.close()
            print("Producer closed.")

if __name__ == "__main__":
    run_producer()