# makur, utils

import glob
import os
import random


import numpy as np

import io_util
# import das_util


def shuffle_indices(indice):
    indice_tuple_list = []
    for k in indice.keys():
        for v in indice[k]:
            indice_tuple_list.append((k, v))
    # data_num = len(indice_tuple_list)
    random.shuffle(indice_tuple_list)
    return indice_tuple_list


def convert_str(num, length_to_write):
    if 10**length_to_write > num:
        num_str = str(num)
        if len(num_str) > length_to_write:
            ret_str = num_str[:length_to_write:1]
        else:
            ret_str = num_str
            while len(ret_str) < length_to_write:
                ret_str = '0'+ret_str
    else:
        num_str = str(num)
        ret_str = num_str
    return ret_str


def save_txt(txt_path, message_str):
    txt_file = open(txt_path, 'w')
    txt_file.write(message_str)
    txt_file.close()


def get_features_acc_channel(data_folder, channel_idx, extension='.npy'):
    channel_idx_folder = data_folder + str(channel_idx)+'\\'
    file_paths = io_util.get_files(channel_idx_folder, extension)
    all_features = None
    for file_path in file_paths:
        cur_features = np.load(file_path)
        if all_features is None:
            all_features = cur_features
        else:
            all_features = np.concatenate((all_features, cur_features), axis=1)
    return all_features


def cur_directory():
    return os.getcwd()


def get_window(n_window, window_method=None):
    window = np.zeros(n_window)
    if window_method == 'hann':
        for n in range(n_window):
            window[n] = 0.5 - 0.5*np.cos(2*np.pi*n/(n_window-1))
    elif window_method == 'hamming':
        for n in range(n_window):
            window[n] = 0.54 - 0.46*np.cos(2*np.pi*n/(n_window-1))
    else:
        window = np.ones(n_window)
    return window


def get_specific_feature_data(data_folder, model_folder, ch_idx, data_idx, samples_num_per_record, norm_str, extension):
    ch_idx = int(ch_idx)
    data_idx = int(data_idx)
    file_idx = data_idx // samples_num_per_record
    feature_idx = data_idx % samples_num_per_record
    ch_idx_folder = data_folder + str(ch_idx) + '\\'
    file_names = io_util.get_files(ch_idx_folder, extension)
    record_name = file_names[file_idx]
    data_path = ch_idx_folder + record_name
    cur_data = np.load(data_path)
    if norm_str == '01':
        features_min_path = model_folder + 'features_min.npy'
        features_min = np.load(features_min_path)

        features_max_path = model_folder + 'features_max.npy'
        features_max = np.load(features_max_path)
        features_range = features_max - features_min

        cur_feature_norm = ((cur_data[::, feature_idx] - features_min) / features_range).reshape(-1, 1)
        cur_feature_norm[cur_feature_norm > 1.0] = 1.0
        cur_feature_norm[cur_feature_norm < 0.0] = 0.0
    elif norm_str == 'z_score':
        features_mean_path = model_folder + 'features_mean.npy'
        features_mean = np.load(features_mean_path)

        features_std_path = model_folder + 'features_std.npy'
        features_std = np.load(features_std_path)

        cur_feature_norm = ((cur_data[::, feature_idx] - features_mean) / features_std).reshape(-1, 1)
    else:
        print('Unknown normalization method')
        raise (ValueError('Normalization Method', norm_str, ' is not valid!'))
    return cur_feature_norm


def get_specific_raw_data_segment(data_folder, ch_idx, data_idx, samples_num_per_record, raw_data_extension):
    ch_idx = int(ch_idx)
    data_idx = int(data_idx)
    file_idx = data_idx // samples_num_per_record
    feature_idx = data_idx % samples_num_per_record
    file_names = io_util.get_files(data_folder, raw_data_extension)
    record_name = file_names[file_idx]
    data_path = data_folder + record_name
    cur_data = das_util.read_raw_data(data_path, ch_idx, ch_idx+1, hp.raw_data_time_start_idx, hp.raw_data_time_end_idx,
                                      hp.raw_data_channel_num, hp.raw_data_header_bytes, hp.raw_data_chunk_heigth)
    feature_start_idx = int(feature_idx*hp.n_window)
    feature_end_idx = int(feature_start_idx+hp.n_window)
    cur_data_segment = cur_data[feature_start_idx:feature_end_idx:1, 0]
    return cur_data_segment


def calc_features(cur_chunk, window, bp_low_freq_idx, bp_high_freq_idx, feature_num, nfft):
    cur_chunk_windowed = np.multiply(cur_chunk, window)
    cur_freqs = np.fft.fft(cur_chunk_windowed, n=nfft, axis=0)
    cur_freqs_mag = np.abs(cur_freqs[bp_low_freq_idx:bp_high_freq_idx])

    cur_freqs_mag += 1e-7  # prevent division by zero
    cur_freqs_mag_norm = 10*np.log10(cur_freqs_mag)
    # with np.errstate(divide='raise', invalid='raise'):
    #     try:
    #         cur_freqs_mag_norm = cur_freqs_mag / np.sum(cur_freqs_mag)
    #     except:
    #         cur_freqs_mag_norm = np.ones(feature_num) / feature_num
    # cur_freqs_mag_norm = 10 * np.log10(cur_freqs_mag_norm)
    return cur_freqs_mag_norm.astype(np.float32)


def normalize_features(feature_in, model_folder, norm_method):
    if norm_method == '01':
        features_min_path = model_folder + 'features_min.npy'
        features_min = np.load(features_min_path)

        features_max_path = model_folder + 'features_max.npy'
        features_max = np.load(features_max_path)
        features_range = features_max - features_min

        cur_feature_norm = ((feature_in - features_min) / features_range).reshape(-1, 1)
        cur_feature_norm[cur_feature_norm > 1.0] = 1.0
        cur_feature_norm[cur_feature_norm < 0.0] = 0.0
    elif norm_method == 'z_score':
        features_mean_path = model_folder + 'features_mean.npy'
        features_mean = np.load(features_mean_path)

        features_std_path = model_folder + 'features_std.npy'
        features_std = np.load(features_std_path)
        cur_feature_norm = ((feature_in - features_mean) / features_std).reshape(-1, 1)
    else:
        print('Unknown normalization method')
        raise (ValueError('Normalization Method', norm_method, ' is not valid!'))
    return cur_feature_norm


def get_file_path(data_folder, data_idx, samples_num_per_record, extension):
    data_idx = int(data_idx)
    file_idx = data_idx // samples_num_per_record
    # feature_idx = data_idx % samples_num_per_record
    file_names = io_util.get_files(data_folder, extension)
    record_name = file_names[file_idx]
    data_path = data_folder + record_name
    return data_path
