import tensorflow as tf
import tensorflow_io as tfio

LABELS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']


def get_audio_and_label(filename):
    # TODO: Write your code here
    pass
    

def get_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s):
    # TODO: Write your code here
    pass


def get_log_mel_spectrogram(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency):
    # TODO: Write your code here
    pass


def get_mfccs(filename, downsampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_coefficients):
    # TODO: Write your code here
    pass
