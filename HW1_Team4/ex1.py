import sounddevice as sd
from scipy.io.wavfile import write
import time as tm  # used to get the timestamp used as filename
import os
import argparse as ap
import tensorflow as tf
import tensorflow_io as tfio
# from preprocessing import *

# TO-DO remove all arguments except for device
parser = ap.ArgumentParser()
parser.add_argument('--resolution', type=str, default='int16')
parser.add_argument('--sampleRate', type=int, default=16000)
parser.add_argument('--nChannels', type=int, default=1)
parser.add_argument('--duration', type=int, default=1)
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

#is_silence hyperparametrs
sampling_rate = 16000
downsampling_rate = 16000
frame_length_in_s = 0.016
dbFSthresh = -140 
duration_time = 0.128

# preprocessing funtion
def get_audio_from_numpy(indata):
    indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1  # CORRECT normalization between -1 and 1
    indata = tf.squeeze(indata)

    return indata


# you modified the get_sepec func. - added indata
def get_spectrogram_from_numpy(indata, sampling_rate, downsampling_rate, frame_length_in_s, frame_step_in_s):
    indata = get_audio_from_numpy(indata)

    if downsampling_rate != sampling_rate:
        sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
        audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)

    sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)

    # zero padding
    zero_padding = tf.zeros(tf.cast(downsampling_rate, dtype=tf.int32) - tf.shape(indata), dtype=tf.float32)
    audio_padded = tf.concat([indata, zero_padding], axis = 0)

    # conversion from samples to number of seconds
    frame_length = int(frame_length_in_s * sampling_rate_float32)
    frame_step = int(frame_step_in_s * sampling_rate_float32)

    # short time fourier transform
    spectrogram = stft = tf.signal.stft(
        audio_padded, 
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length
    )
    spectrogram = tf.abs(stft)

    return spectrogram
# end of preprocessing


# VAD 
def is_silence(indata, sampling_rate, downsampling_rate, frame_length_in_s, dbFSthresh, duration_time):
    print("issilence called")
    spectrogram = get_spectrogram_from_numpy(
        indata,
        sampling_rate,
        downsampling_rate,
        frame_length_in_s,
        frame_length_in_s
    )
    dbFS = 20 * tf.math.log(spectrogram + 1.e-6)
    energy = tf.math.reduce_mean(dbFS, axis=1)
    non_silence = energy > dbFSthresh
    non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
    non_silence_duration = (non_silence_frames + 1) * frame_length_in_s

    if non_silence_duration > duration_time:
        return 0
    else:
        return 1


# function executed in parallel with the recording
# indata is a numpy array
# freames is the number of sample inside indata
def callback(indata, frames, time, status):
    global store_audio
    global args

    if store_audio is True:
        if is_silence(indata, sampling_rate, downsampling_rate, frame_length_in_s, dbFSthresh, duration_time) == 0:  # if not silence
            timestamp = tm.time()
            write(f'.\\hw01\\recordings\\{timestamp}.wav', args.sampleRate, indata)
            print("audio saved")
    

print("recording started")

store_audio = True

# device=0 -> default microphone
# channels=1 -> we use only one channel (there are multichannel microphones)
# all the code inside the with block is executed while the stream is open
# blocksize define the number of sample after which the callback function is collecd
with sd.InputStream(device=args.device, channels=args.nChannels, samplerate=args.sampleRate, dtype=args.resolution, callback=callback, blocksize=args.duration*args.sampleRate):  # create an input stream
    while True:  # keep the stream open
        key = input()
        if key in ['q', 'Q']:  # stop recording if q is pressed
            print("exit")
            break
        if key in ['p', 'P']:
            store_audio = not store_audio
            print(f"store audio: {store_audio}")




