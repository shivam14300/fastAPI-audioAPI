import librosa
import sklearn
import scipy
import json
import csv
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt
from numpy.fft import fft, ifft
# from control import TransferFunction, forced_response
# import matplotlib.pyplot as plt
import librosa.display
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def extract_features(wav_file_path):
    if not isinstance(wav_file_path, str):
        return None
    features = {}
    print(wav_file_path)
    x, fs = librosa.load(wav_file_path, sr=8000)
    # x is the amplitude of the audio

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    # Taking time samples for the given duration
    duration = 5.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs

    # Envelope Detection..................................................................................
    x = x[0: len(t)]
    signal = x
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    nyq = 0.5 * fs
    cutoff = 1000
    order = 3
    n = samples
    normalised_cutoff = cutoff / nyq
    # getting filter coefficients
    b, a = butter(order, normalised_cutoff, btype='low', analog=False)
    filtered_envelope = filtfilt(b, a, amplitude_envelope)
    features['envelope_mean'] = np.mean(filtered_envelope)
    features['envelope_std'] = np.std(filtered_envelope)

    # Doing post feature processing using Savitzky-Golay filter............................................
    # filter order = 3, size = 11
    denoised_output = scipy.signal.savgol_filter(filtered_envelope, 11, 3)
    # Downsampling the denoised signal to the below sampling rate
    fs2 = 8000
    downsampled_samples = int(fs2 * duration)
    t2 = np.arange(downsampled_samples) / fs2
    downsampled_output = scipy.signal.resample(denoised_output, downsampled_samples)

    # Now extracting spectral features...................................................................
    # spectral centroids
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=fs2)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    features['spectral_centroid_range'] = np.amax(spectral_centroids) - np.amin(spectral_centroids)
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_max'] = np.amax(spectral_centroids)
    features['spectral_centroid_min'] = np.amin(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)

    # spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=fs2)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)

    # spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr=fs2)[0]  # spectral BW of order = 2
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

    # ZCR - calculating number of zero crossings in all samples from 0 to 5 seconds
    zero_crossings = librosa.zero_crossings(x[0:(fs2 * 5)], pad=False)
    features['ZCR_5'] = sum(zero_crossings)

    mfccs = librosa.feature.mfcc(x, sr=fs2)
    # Taking the time average of the MFCCs:
    mfccs_mean = np.zeros((len(mfccs),))
    for i in range(len(mfccs)):
        mfccs_mean[i] = np.mean(mfccs[i])
        key = "MFCC_" + str(i + 1)
        features[key] = mfccs_mean[i]

    return features

"""
file_path = 'C:\\Users\\jtuli\\Desktop\\Spirometry App\\Respiratory_Sound_Database\\' \
            'Respiratory_Sound_Database\\sound_base\\'
conditions_file_path = 'C:\\Users\\jtuli\\Desktop\\Spirometry App\\Respiratory_Sound_Database\\' \
            'Respiratory_Sound_Database\\patient_diagnosis.csv'

extracted_features = {}
ids = list(range(101, 223))
# classes contains the lung health conditions corresponding to patient id
classes = {}

with open(conditions_file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        classes[int(row[0])] = row[1]

for i in ids:
    audio_data = file_path + str(i) + ".wav"
    extracted_features[i] = extract_features(audio_data)

# Storing the training data
data_dict = {}
i = 0
for patient_id in extracted_features:
    patient_data = {}
    patient_data['lung_condition'] = classes[patient_id]
    patient_data['features'] = extracted_features[patient_id]
    data_dict[patient_id] = patient_data

with open('training_data.json', 'w') as file:
    json_string = json.dumps(data_dict, cls=NpEncoder)
    file.write(json_string)
"""

