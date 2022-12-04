import tensorflow as tf
import tensorflow_io as tfio
import sounddevice as sd
from scipy.io.wavfile import write
import argparse as ap
import numpy as np
import os
import zipfile
import psutil
import uuid  # to retrieve the mac address
import redis
from time import time as tm, sleep

parser = ap.ArgumentParser()
parser.add_argument('--resolution', type=str, default='int16')  # debug
parser.add_argument('--sampleRate', type=int, default=16000)  # debug
parser.add_argument('--nChannels', type=int, default=1)  # debug
parser.add_argument('--duration', type=int, default=1)  # debug
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--host', type=str, default="redis-18937.c72.eu-west-1-2.ec2.cloud.redislabs.com")
parser.add_argument('--port', type=int, default=18937)
parser.add_argument('--user', type=str, default="default")
parser.add_argument('--password', type=str, default="DlfmUPWr2iMKAbzvEiwzLCizwt2yjLkP")
parser.add_argument('--delete', type=int, default=0)  # debug
parser.add_argument('--verbose', type=int, default=0)  # debug

args = parser.parse_args()


class SmartBatteryMonitoring:
    def __init__(self, ISSILENCE_ARGS, PREPROCESSING_ARGS, LABELS, model_path) -> None:
        ######################################
        # -- VAD and classification model -- #
        ######################################
        self.ISSILENCE_ARGS = ISSILENCE_ARGS
        self.PREPROCESSING_ARGS = PREPROCESSING_ARGS
        self.LABELS = LABELS

        self.model_path = model_path
        self.zip_model_path = f'{self.model_path}.zip'

        self.store_audio = False

        self.downsampling_rate = PREPROCESSING_ARGS['downsampling_rate']
        self.sampling_rate = self.downsampling_rate
        self.sampling_rate_int64 = tf.cast(self.downsampling_rate, tf.int64)
        self.frame_length = int(self.downsampling_rate * PREPROCESSING_ARGS['frame_length_in_s'])
        self.frame_step = int(self.downsampling_rate * PREPROCESSING_ARGS['frame_step_in_s'])
        self.num_mel_bins = PREPROCESSING_ARGS['num_mel_bins']
        self.num_spectrogram_bins = self.frame_length // 2 + 1
        self.lower_frequency = PREPROCESSING_ARGS["lower_frequency"]
        self.upper_frequency = PREPROCESSING_ARGS["upper_frequency"]
        self.num_coefficients = PREPROCESSING_ARGS["num_coefficients"]

        # compute the linear_to_mel_weight_matrix one only one time
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=self.num_spectrogram_bins,
            sample_rate=self.downsampling_rate,
            lower_edge_hertz=self.lower_frequency,
            upper_edge_hertz=self.upper_frequency
        )

        # load model
        self.interpreter, self.input_details, self.output_details = self.import_tflite_model()

        ##########################
        # -- REDIS timeseries -- #
        ##########################
        self.REDIS_HOST = args.host
        self.REDIS_PORT = args.port
        self.REDIS_USER = args.user
        self.REDIS_PASSWORD = args.password

        # connect to redis
        self.redis_client = self.connect_to_redis()

        self.battery_string, self.power_string, self.power_plugged_seconds_string = self.generate_timeseries_name()

        # create timeseries
        self.create_timeseries()

        self.modifiy_retention_periods()

    def connect_to_redis(self):
        # establish a connection to redis
        redis_client = redis.Redis(host=self.REDIS_HOST, port=self.REDIS_PORT,
                                   username=self.REDIS_USER, password=self.REDIS_PASSWORD)
        if args.verbose == 1:
            print('Is connected: ', redis_client.ping())

        return redis_client

    def generate_timeseries_name(self):
        mac_address = hex(uuid.getnode())
        battery_string = f'{mac_address}:battery'
        power_string = f'{mac_address}:power'
        power_plugged_seconds_string = f'{mac_address}:plugged_seconds'

        return battery_string, power_string, power_plugged_seconds_string

    def create_timeseries(self):
        # bucket size duration for the plugged_seconds timeseries
        bucket_duration_in_ms = 24 * 60 * 60 * 1000  # 24h
        # bucket_duration_in_ms = 60 * 1000  # 60s for testing
        if args.verbose == 1:
            print(f"bucket duration: {bucket_duration_in_ms}ms")

        # delete previous timeseries
        if args.delete == 1:
            self.redis_client.delete(self.battery_string)
            self.redis_client.delete(self.power_string)
            self.redis_client.delete(self.power_plugged_seconds_string)

        # avoid errors if timeseries is already created
        try:
            # create a timeseries
            self.redis_client.ts().create(self.battery_string)  # default chunk_size = 4KB
            self.redis_client.ts().create(self.power_string)
            # create timeseries and rule for counting the seconds the power is plugged every 24h
            self.redis_client.ts().create(self.power_plugged_seconds_string,
                                          chunk_size=128)  # one record each day -> we can reduce chunk_size
            self.redis_client.ts().createrule(self.power_string, self.power_plugged_seconds_string,
                                              aggregation_type='sum', bucket_size_msec=bucket_duration_in_ms)
        except redis.ResponseError:
            pass  # do nothing is the timeseries is already created

    def modifiy_retention_periods(self):
        # retention periods
        battery_retention = int(5 * ((2 ** 20) / 1.6) * 1000)  # 3276800000 ms
        power_retention = int(5 * ((2 ** 20) / 1.6) * 1000)
        power_plugged_seconds_retention = int(((2 ** 20) / 1.6) * 24 * 60 * 60 * 1000)  # 5.6623104e13 ms

        if args.verbose == 1:
            print(f"retention period battery: {battery_retention}ms")
            print(f"retention period power: {power_retention}ms")
            print(f"retention period plugged seconds: {power_plugged_seconds_retention}ms")

        # create retention window
        self.redis_client.ts().alter(self.battery_string, retention_msec=battery_retention)
        self.redis_client.ts().alter(self.power_string, retention_msec=power_retention)
        self.redis_client.ts().alter(self.power_plugged_seconds_string, retention_msec=power_plugged_seconds_retention)

    def add_value_to_timeseries(self):
        timestamp_ms = int(tm() * 1000)

        # retreive info about battery and power
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)

        self.redis_client.ts().add(self.battery_string, timestamp_ms, battery_level)
        self.redis_client.ts().add(self.power_string, timestamp_ms, power_plugged)

        if args.verbose == 1:
            print('timestamp: ', timestamp_ms)
            print(self.battery_string, " - ", battery_level)
            print(self.power_string, " - ", power_plugged)

    def preprocessing(self, indata, sampling_rate):
        # PRE_PROCESSING (LOG-MEL SPECTOGRAM)
        audio = tf.convert_to_tensor(indata, dtype=tf.float32)

        audio = tf.squeeze(audio)  # remove axis for multiple channel audio (we use single input)

        zero_padding = tf.zeros(sampling_rate - tf.shape(audio), dtype=tf.float32)
        audio_padded = tf.concat([audio, zero_padding], axis=0)

        if self.downsampling_rate != sampling_rate:
            audio_padded = tfio.audio.resample(audio_padded, self.sampling_rate_int64, self.downsampling_rate)

        stft = tf.signal.stft(
            audio_padded,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )

        spectrogram = tf.abs(stft)

        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        # PRE_PROCESSING (MFCCs SPECTOGRAM)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.num_coefficients]

        return mfccs

    def inference(self, mfccs):
        # conv layer input has 4 dimension: B, T, F, C
        # T, F given by log_mel_spectogram
        # B = 1 single batch for inference
        # C = 1 single channel
        # we have to add the B and C axis
        mfccs = tf.expand_dims(mfccs, 0)  # batch axis
        mfccs = tf.expand_dims(mfccs, -1)  # channel axis

        # mfccs = tf.image.resize(log_mel_spectrogram, [32, 32])  # used only for spectrogram

        self.interpreter.set_tensor(self.input_details[0]['index'], mfccs)  # pass the input values to the model
        self.interpreter.invoke()  # inference
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # only one batch during inference
        return output[0]

    def unzip_model(self):
        # un-zip model if not already done
        if not os.path.isfile(self.model_path):
            print("un-zipping model..")
            # un-zip model
            with zipfile.ZipFile(self.zip_model_path, 'r') as zip_ref:
                zip_ref.extractall()
            print(f'{self.zip_model_path} un-zipped!')

    def get_audio_from_numpy(self, indata):
        indata = tf.convert_to_tensor(indata, dtype=tf.float32)
        indata = 2 * ((indata + 32768) / (32767 + 32768)) - 1  # CORRECT normalization between -1 and 1
        indata = tf.squeeze(indata)

        return indata

    def get_spectrogram_from_numpy(self, indata, sampling_rate, downsampling_rate, frame_length_in_s, frame_step_in_s):
        indata = self.get_audio_from_numpy(indata)

        sampling_rate_float32 = tf.cast(downsampling_rate, tf.float32)

        # zero padding
        zero_padding = tf.zeros(tf.cast(downsampling_rate, dtype=tf.int32) - tf.shape(indata), dtype=tf.float32)
        audio_padded = tf.concat([indata, zero_padding], axis=0)

        if downsampling_rate != sampling_rate:
            sampling_rate_int64 = tf.cast(sampling_rate, tf.int64)
            audio_padded = tfio.audio.resample(audio_padded, sampling_rate_int64, downsampling_rate)

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

    # VAD
    def is_silence(self, indata, sampling_rate, downsampling_rate, frame_length_in_s, dbFSthresh, duration_time):
        if args.verbose == 1:
            print("issilence called")

        spectrogram = self.get_spectrogram_from_numpy(
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

    def import_tflite_model(self):
        # import the model using the tflite interpreter

        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()  # instanciate the model
        # it reserves the memory for the tensor
        # in keras is done automatically (because in the cloud you have a lot o memory)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details

    # function executed in parallel with the recording
    # indata is a numpy array
    # frames is the number of sample inside indata
    def callback(self, indata, frames, time, status):
        if self.store_audio is True:
            # save battery values to redis timeseries
            self.add_value_to_timeseries()

        if self.is_silence(indata, **self.ISSILENCE_ARGS) == 0:  # if not silence
            # preprocess recorded audio
            mfccs = self.preprocessing(indata, frames)
            # predict label
            output = self.inference(mfccs)
            if args.verbose == 1:
                print("audio saved")
            for index, prob in enumerate(output):
                if args.verbose == 1:
                    print(f'Probability of {self.LABELS[index]} = {prob * 100:.2f}')
                if prob > 0.95:
                    if index == 0:  # stop
                        print("battery monitoring stopped")
                        self.store_audio = False
                    else:  # go
                        print("battery monitoring started")
                        self.store_audio = True

    def start_recording(self):
        print("recording started")

        # device=0 -> default microphone
        # channels=1 -> we use only one channel (there are multichannel microphones)
        # all the code inside the with block is executed while the stream is open
        # blocksize define the number of sample after which the callback function is collecd
        with sd.InputStream(device=args.device, channels=args.nChannels, samplerate=args.sampleRate,
                            dtype=args.resolution, callback=self.callback,
                            blocksize=args.duration * args.sampleRate):  # create an input stream
            while True:  # keep the stream open
                key = input()
                if key in ['q', 'Q']:  # stop recording if q is pressed
                    print("exit")
                    break
                if key in ['p', 'P']:
                    self.store_audio = not self.store_audio
                    print(f"store audio: {self.store_audio}")


def main():
    # model hyperparameters
    PREPROCESSING_ARGS = {
        'downsampling_rate': 16000,
        'frame_length_in_s': 0.016,
        'frame_step_in_s': 0.016,
        'num_mel_bins': 10,
        'lower_frequency': 20,
        'upper_frequency': 8000,
        'num_coefficients': 40
    }

    # is_silence hyperparametrs
    ISSILENCE_ARGS = {
        'sampling_rate': 16000,
        'downsampling_rate': 16000,
        'frame_length_in_s': 0.016,
        'dbFSthresh': -140,
        'duration_time': 0.128
    }

    # order of the labels must be the same as the one used in training
    LABELS = ['stop', 'go']  # ATTENTION, MUST BE SAME ORDER

    model_path = os.path.join('.', 'tflite_models', 'model4.tflite')

    s = SmartBatteryMonitoring(ISSILENCE_ARGS, PREPROCESSING_ARGS, LABELS, model_path)
    s.unzip_model()
    # s.store_audio = True
    s.start_recording()


if __name__ == "__main__":
    main()
