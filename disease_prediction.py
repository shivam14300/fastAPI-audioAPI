# Extract features for this audio clip
import pickle
import numpy as np
from feature_extraction import extract_features


def pred_lung_health(wav_file_path):
    if not isinstance(wav_file_path, str):
        return None
    features_dict = extract_features(wav_file_path)
    feature_vector = np.zeros((1, len(features_dict)))
    temp_list = []
    for feature in features_dict:
        temp_list.append(features_dict[feature])
    feature_vector[0] = np.array(temp_list)

    # open model
    classifier = pickle.load(open('gradient_boosting_classifier.sav', 'rb'))
    # Predicting lung health
    predicted_condition = classifier.predict(feature_vector)

    return predicted_condition[0]


# wav_path = 'C:\\Users\\jtuli\\Desktop\\Spirometry App\\Respiratory_Sound_Database\\' \
#             'Respiratory_Sound_Database\\sound_base\\' + '120.wav'
# pred_condition = pred_lung_health(wav_path)
# print(pred_condition)