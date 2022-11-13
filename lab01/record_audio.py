import sounddevice as sd
from scipy.io.wavfile import write
from time import time  # used to get the timestamp used as filename
import os
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--resolution', type=str, default='int32')
parser.add_argument('--sampleRate', type=int, default=48000)
parser.add_argument('--nChannels', type=int, default=1)
parser.add_argument('--duration', type=int, default=1)

args = parser.parse_args()

# function executed in parallel with the recording
# indata is a numpy array
# freames is the number of sample inside indata
def callback(indata, frames, time, status):
    global store_audio
    global args

    if store_audio is True:
        timestamp = time()
        write(f'{timestamp}.waw', args.sampleRate, indata)
        size_in_bytes = os.path.getsize(f'{timestamp}.wav')
        size_in_kb = size_in_bytes  / 1024.
        print(f'Size {size_in_kb} KB')
    

print("recording started")

store_audio = True

# device=0 -> default microphone
# channels=1 -> we use only one channel (there are multichannel microphones)
# all the code inside the with block is executed while the stream is open
# blocksize define the number of sample after which the callback function is collecd
with sd.InputStream(device=0, channels=args.nChannels, samplerate=args.sampleRate, dtype=args.resolution, callback=callback, blocksize=args.duration*args.sampleRate):  # create an input stream
    while True:  # keep the stream open
        key = input()
        if key in ['q', 'Q']:  # stop recording if q is pressed
            print("recording started")
            break
        if key in ['p', 'P']:
            store_audio = not store_audio


