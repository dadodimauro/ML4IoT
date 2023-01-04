"""
This code shows how to store and retrieve JSON data in Redis.
"""

import redis
import json
import uuid  # used to crate unique identifier, needed to store a json file in REDIS
from redis.commands.json.path import Path


REDIS_HOST = "redis-18937.c72.eu-west-1-2.ec2.cloud.redislabs.com"
REDIS_PORT = 18937
REDIS_USER = "default"
REDIS_PASSWORD = "DlfmUPWr2iMKAbzvEiwzLCizwt2yjLkP"

# Connect to Redis server
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USER, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

# Flush DB (if needed)
# redis_client.flushdb()

# Add some JSON objects to the index
redis_client.json().set(f'todo:{uuid.uuid4()}', Path.root_path(), {'message': 'Buy milk', 'completed': True})
redis_client.json().set(f'todo:{uuid.uuid4()}', Path.root_path(), {'message': 'Do homework 2', 'completed': True})
redis_client.json().set(f'todo:{uuid.uuid4()}', Path.root_path(), {'message': 'Do homework 3', 'completed': False})

# Get all objects
print('Getting all objects...')
keys = redis_client.keys('todo:*')

for key in keys:
    key = key.decode()
    item = redis_client.json().get(key)
    print(key, item)
