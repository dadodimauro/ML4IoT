from time import time, sleep
import paho.mqtt.client as mqtt
import psutil
import uuid  # to retrieve the mac address
import argparse as ap
import json

parser = ap.ArgumentParser()
parser.add_argument('--topic', type=str, default="s306089")
parser.add_argument('--verbose', type=int, default=0)  # debug
parser.add_argument('--laptop', type=int, default=1)  # debug

args = parser.parse_args()


def create_message():
    mac_address = hex(uuid.getnode())
    timestamp = int(time() * 1000)  # timestamp in ms
    if args.laptop == 1:
        battery_level = psutil.sensors_battery().percent
        power_plugged = psutil.sensors_battery().power_plugged
    else:  # on DESKTOP since psutils.sensors is not working
        battery_level = 100
        power_plugged = True

    message_dict = {
        "mac_address": mac_address,
        "timestamp": timestamp,
        "battery_level": battery_level,
        "power_plugged": power_plugged
    }

    # Encode a dict to JSON string
    message_string = json.dumps(message_dict)

    return message_string


# Create a new MQTT client
client = mqtt.Client()

# Connect to the MQTT broker
client.connect('mqtt.eclipseprojects.io', 1883)  # 'mqtt.eclipseprojects.io' is a free public broker used for testing
                                                 # only the broker knows the REDIS acces credential, the client
                                                 # doesn't know ads a layer of security
                                                 # (still all the message are NOT encripted in MQTT)

# Publish a message to a topic
for i in range(10):
    message_string = create_message()

    if args.verbose == 1:
        print(message_string)

    client.publish(args.topic, message_string)
    sleep(1)
