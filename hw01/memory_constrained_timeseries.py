import argparse as ap
import redis
from time import time, sleep
from datetime import date, datetime
import psutil
import uuid  # to retrieve the mac address

parser = ap.ArgumentParser()
parser.add_argument('--host', type=str, default="redis-18937.c72.eu-west-1-2.ec2.cloud.redislabs.com")
parser.add_argument('--port', type=int, default=18937)
parser.add_argument('--user', type=str, default="default")
parser.add_argument('--password', type=str, default="DlfmUPWr2iMKAbzvEiwzLCizwt2yjLkP")
parser.add_argument('--delete', type=int, default=0)  # debug
parser.add_argument('--verbose', type=int, default=0)  # debug

args = parser.parse_args()

REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USER = args.user
REDIS_PASSWORD = args.password

# establish a connection to redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USER, password=REDIS_PASSWORD)

if args.verbose == 1:
    print('Is connected: ', redis_client.ping())

mac_address = hex(uuid.getnode())
battery_string = f'{mac_address}:battery'
power_string = f'{mac_address}:power'
power_plugged_seconds_string = f'{mac_address}:plugged_seconds'

# print(power_plugged_seconds_string)

# bucket size duration for the plugged_seconds timeseries
bucket_duration_in_ms = 24 * 60 * 60 * 1000  # 24h
# bucket_duration_in_ms = 60 * 1000  # 60s for testing

# delete previous timeseries
if args.delete == 1:
    redis_client.delete(battery_string)
    redis_client.delete(power_string)
    redis_client.delete(power_plugged_seconds_string)

# avoid errors if timeseries is already created
try:
    # create a timeseries
    redis_client.ts().create(battery_string)  # default chunk_size = 4KB
    redis_client.ts().create(power_string)
    # create timeseries and rule for counting the seconds the power is plugged every 24h
    redis_client.ts().create(power_plugged_seconds_string, chunk_size=128)  # one record each day -> we can reduce chunk_size
    redis_client.ts().createrule(power_string, power_plugged_seconds_string, aggregation_type='sum', bucket_size_msec=bucket_duration_in_ms)
except redis.ResponseError:
    pass  # do nothing is the timeseries is already created

# retention periods
battery_retention = int(5 * (2^20) / 1.6 * 1000)  # 3276800000s
power_retention = int(5 * (2^20) / 1.6 * 1000)
power_plugged_seconds_retention = int((2^20) / 1.6 * 1000)  # 655360000s

# create retention window
redis_client.ts().alter(battery_string, retention_msec=battery_retention)
redis_client.ts().alter(power_string, retention_msec=power_retention)
redis_client.ts().alter(power_plugged_seconds_string, retention_msec=power_plugged_seconds_retention)

seconds_counter = 0

while True:
    timestamp_ms = int(time() * 1000)

    # increment seconds counter
    seconds_counter += 1

    # retreive info about battery and power
    battery_level = psutil.sensors_battery().percent
    power_plugged = int(psutil.sensors_battery().power_plugged)

    redis_client.ts().add(battery_string, timestamp_ms, battery_level)
    redis_client.ts().add(power_string, timestamp_ms, power_plugged)

    if args.verbose == 1:
        print(f"runnig for {seconds_counter} seconds")
        print(battery_string, " - ", battery_level)
        print(power_string, " - ", power_plugged)

    sleep(1)