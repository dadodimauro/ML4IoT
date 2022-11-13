import redis
from time import time
from time import sleep
from datetime import date, datetime

REDIS_HOST = "redis-18937.c72.eu-west-1-2.ec2.cloud.redislabs.com"
REDIS_PORT = 18937
REDIS_USER = "default"
REDIS_PASSWORD = "DlfmUPWr2iMKAbzvEiwzLCizwt2yjLkP"

# establish a connection to redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USER, password=REDIS_PASSWORD)

# politecnico wifi block redis connections
print('Is connected: ', redis_client.ping())

# delete previous timeseries
redis_client.delete('temperature')
# redis_client.delete('temperature_uncompressed')
redis_client.delete('temperature_avg')

# average the data of temperature ts in buckets of 1000ms
bucket_duration_in_ms = 1000

# avoid errors if timeseries is already created
try:
    # create a timeseries
    redis_client.ts().create('temperature', chunk_size=128)  # with chunk size we can control the dimension oh the chunks of the database
    # create a timeseries for the average
    redis_client.ts().create('temperature_avg')
    redis_client.ts().createrule('temperature', 'temperature_avg', aggregation_type='avg', bucket_size_msec=bucket_duration_in_ms)
except redis.ResponseError:
    pass  # do nothing is the timeseries is already created

one_day_in_ms = 24 * 60 * 60 * 1000
# create a retention window in redis
redis_client.ts().alter('temperature', retention_msec=one_day_in_ms)

# add 100 values to the timeseries
for i in range(100):
    timestamp_ms = int(time() * 1000)
    redis_client.ts().add('temperature', timestamp_ms, 25 + i // 50)
    sleep(0.1)

# print some statistics from redis
print('===TEMPERATURE INFO===')
print("memory (bytes): ", redis_client.ts().info('temperature').memory_usage)
print("# samples: ", redis_client.ts().info('temperature').total_samples)
print("# chunks: ", redis_client.ts().info('temperature').chunk_count)

# print some statistics from redis
print('===AVG INFO===')
print("memory (bytes): ", redis_client.ts().info('temperature_avg').memory_usage)
print("# samples: ", redis_client.ts().info('temperature_avg').total_samples)
print("# chunks: ", redis_client.ts().info('temperature_avg').chunk_count)

# # create uncompressed timeseries
# # avoid errors if timeseries is already created
# try:
#     # create a timeseries
#     redis_client.ts().create('temperature_uncompressed', chunk_size=128, uncompressed=True)  # with chunk size we can control the dimension oh the chunks of the database
# except redis.ResponseError:
#     pass  # do nothing is the timeseries is already created

# # add 100 values to the timeseries
# for i in range(100):
#     timestamp_ms = int(time() * 1000)
#     redis_client.ts().add('temperature_uncompressed', timestamp_ms, 25 + i // 50)
#     sleep(0.1)

# # print some statistics from redis
# print('===INFO===')
# print("memory (bytes): ", redis_client.ts().info('temperature_uncompressed').memory_usage)
# print("# samples: ", redis_client.ts().info('temperature_uncompressed').total_samples)
# print("# chunks: ", redis_client.ts().info('temperature_uncompressed').chunk_count)






