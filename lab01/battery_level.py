from datetime import datetime
import psutil
import uuid  # to retrieve the mac address
from time import sleep, time
from datetime import datetime

mac_address = hex(uuid.getnode())

while True:
    timestamp = time()  # number of seconds from 00:00 - 01/01/1970 
    formatted_datetime = datetime.fromtimestamp(timestamp)

    battery_level = psutil.sensors_battery().percent
    power_plugged = int(psutil.sensors_battery().power_plugged)
    
    print(f'{formatted_datetime} {mac_address}:battery = {battery_level}')
    print(f'{formatted_datetime} {mac_address}:power = {power_plugged}')

