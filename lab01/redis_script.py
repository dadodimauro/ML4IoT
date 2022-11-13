import redis
from time import time
from datetime import date, datetime

REDIS_HOST = "redis-18937.c72.eu-west-1-2.ec2.cloud.redislabs.com"
REDIS_PORT = 18937
REDIS_USER = "default"
REDIS_PASSWORD = "DlfmUPWr2iMKAbzvEiwzLCizwt2yjLkP"

# establish a connection to redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USER, password=REDIS_PASSWORD)

# politecnico wifi block redis connections
print('Is connected: ', redis_client.ping())

# send data to server
redis_client.set("message", "Hello World!")  # key, value

print(redis_client.get("message").decode())  # decode is used to convert from bit format to string

# avoid errors if timeseries is already created
try:
    # create a timeseries
    redis_client.ts().create('battery')
except redis.ResponseError:
    pass  # do nothing is the timeseries is already created


timestamp = time()
timestamp = int(timestamp * 1000)  # timestamp is required to be in millisecond in redis
# add records to the timeseries
redis_client.ts().add('battery', timestamp, 70)

# create 2 timestamp to read data from a window
# 2022-10-18 14:30:00 -> timestamp
from_datetime = datetime.fromisoformat('2022-10-18 14:30:00')
from_timestamp_in_s = from_datetime.timestamp()
from_timestamp_in_ms = int(from_timestamp_in_s * 1000)
print(from_timestamp_in_ms)

# 2022-10-18 15:00:00 -> timestamp
to_datetime = datetime.fromisoformat('2022-10-18 15:00:00')
to_timestamp_in_s = to_datetime.timestamp()
to_timestamp_in_ms = int(to_timestamp_in_s * 1000)
print(to_timestamp_in_ms)

# read data in a time window
values = redis_client.ts().range('battery', from_timestamp_in_ms, to_timestamp_in_ms)
print(values)

