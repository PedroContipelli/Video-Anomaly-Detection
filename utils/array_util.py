import numpy as np


def interpolate_1D(arr, new_len):
    arr_interpolated = np.interp(np.linspace(0, len(arr), new_len), np.linspace(0, len(arr), len(arr)), arr)
    arr_interpolated[arr_interpolated > 0] = 1
    arr_interpolated[arr_interpolated <= 0] = 0
    return arr_interpolated


def interpolate(features, features_per_video):
    features = np.array(features)
    feature_size = features.shape[1]
    interpolated_features = np.zeros((features_per_video, feature_size))
    interpolation_indicies = np.round(np.linspace(0, len(features) - 1, num=features_per_video + 1))
    count = 0
    for index in range(0, len(interpolation_indicies)-1):
        start = int(interpolation_indicies[index])
        end = int(interpolation_indicies[index + 1])

        assert end >= start

        if start == end:
            temp_vect = features[start, :]
        else:
            temp_vect = np.mean(features[start:end+1, :], axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) == 0:
            print("Error")

        interpolated_features[count,:]=temp_vect
        count = count + 1
    return np.array(interpolated_features)
