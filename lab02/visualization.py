import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from preprocessing import get_audio_and_label
from preprocessing import get_spectrogram
from preprocessing import get_log_mel_spectrogram
from preprocessing import get_mfccs


def visualize_features(filename, downsampling_frequency, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency, num_coefficients):
    audio, _, _ = get_audio_and_label(filename)
    spectrogram, _, _ = get_spectrogram(filename, downsampling_frequency, frame_length_in_s, frame_step_in_s)
    log_mel_spectrogram, _ = get_log_mel_spectrogram(
        filename,
        downsampling_frequency,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    )
    mfccs, _ = get_mfccs(
        filename,
        downsampling_frequency,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        num_coefficients,
    )

    plt.figure(figsize=(15, 5))
    plt.subplot(1,4,1)
    plt.plot(np.arange(audio.shape[0]), audio.numpy())
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Audio')

    plt.subplot(1,4,2)
    log_spectrogram = tf.math.log(spectrogram + 1.e-6)  
    log_spectrogram_vis = tf.transpose(log_spectrogram)
    plt.pcolormesh(log_spectrogram_vis.numpy())
    plt.xlabel('Frame')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')    

    log_mel_spectrogram_vis = tf.transpose(log_mel_spectrogram)
    plt.subplot(1,4,3)
    plt.pcolormesh(log_mel_spectrogram_vis.numpy())
    plt.xlabel('Frame')
    plt.ylabel('Mel bins')
    plt.title('Mel Spectrogram')

    mfccs_vis = tf.transpose(mfccs)
    plt.subplot(1,4,4)
    plt.pcolormesh(mfccs_vis.numpy())
    plt.xlabel('Frame')
    plt.ylabel('Coefficients')
    plt.title('MFCCs')

    plt.show()