# makur, das utils

# import matplotlib.pyplot as plt
import json
import numpy as np
import os


def read_raw_data(file_path, channel_start=0, channel_end=5150, time_start_idx=0, time_end_idx=60*2000,
                  channel_num=5150, header_bytes=0, chunk_height=2*60*2000):
    assert (channel_start >= 0 and channel_end <= 5150 and (channel_end - channel_start) <= channel_num)
    assert (0 <= time_start_idx <= time_end_idx)
    data_size_in_bytes = os.path.getsize(file_path)
    real_iter_num = (data_size_in_bytes-header_bytes)/(channel_num*2*400)
    assert(np.floor(real_iter_num) == real_iter_num)
    time_end_idx = int(min(real_iter_num*400, time_end_idx))
    all_raw_data = np.zeros((0, (channel_end - channel_start)), dtype=np.int16)
    for cur_time_start_idx in range(time_start_idx, time_end_idx, chunk_height):
        offset = header_bytes + cur_time_start_idx * channel_num * 2  # np.int16 is 2 bytes
        count = min((time_end_idx - cur_time_start_idx), chunk_height) * channel_num
        # print('offset: {}, count:{}'.format(offset, count))
        raw_data = np.fromfile(file_path, np.int16, count=count, offset=offset)
        # print('read')
        if raw_data.shape[0] == 0:
            break
        raw_data = raw_data.reshape((-1, channel_num))
        raw_data = raw_data[::, channel_start:channel_end:1]
        all_raw_data = np.concatenate((all_raw_data, raw_data), axis=0)
        print('raw data shape: ', all_raw_data.shape)
    print('raw data read')
    return all_raw_data


def read_norm_power(file_path, channel_start=0, channel_end=5150, time_start_idx=0, time_end_idx=60 * 2000,
                    channel_num=5150, header_bytes=0, chunk_height=60 * 2000):
    assert (channel_start >= 0 and channel_end <= 5150 and (channel_end - channel_start) <= channel_num)
    assert (0 <= time_start_idx <= time_end_idx)
    data_size_in_bytes = os.path.getsize(file_path)
    real_iter_num = (data_size_in_bytes-header_bytes) / (channel_num * 2)
    assert (np.floor(real_iter_num) == real_iter_num)
    time_end_idx = int(min(real_iter_num, time_end_idx))
    all_raw_data = np.zeros((0, (channel_end - channel_start)), dtype=np.float32)
    if header_bytes == 25:
        h_channel_start = np.fromfile(file_path, np.uint16, count=1, offset=3)
        h_channel_end = np.fromfile(file_path, np.uint16, count=1, offset=5)
        h_channel_num = h_channel_end - h_channel_start + 1
        # print(h_channel_start)
        # print(h_channel_end)
        if h_channel_num != channel_num:
            print('Record Channel num is ' + str(h_channel_num))
            return None
    for cur_time_start_idx in range(time_start_idx, time_end_idx, chunk_height):
        offset = header_bytes + cur_time_start_idx * channel_num * 4  # np.float32 is 4 bytes
        count = min((time_end_idx - cur_time_start_idx), chunk_height) * channel_num
        raw_data = np.fromfile(file_path, np.float32, count=count, offset=offset)
        if (offset + count * 4) > data_size_in_bytes:
            raw_data = raw_data[0::1]
        raw_data = raw_data.reshape((-1, channel_num))
        raw_data = raw_data[::, channel_start:channel_end:1]
        all_raw_data = np.concatenate((all_raw_data, raw_data), axis=0)
    return all_raw_data


def read_prob_data(file_path, channel_start=0, channel_end=5150, time_start_idx=0, time_end_idx=60*2000,
                   channel_num=5150, category_num=5, category_idx=1, chunk_height=10*60*60*5):
    assert (channel_start >= 0 and channel_end <= 5150 and (channel_end - channel_start) <= channel_num)
    assert (0 <= time_start_idx <= time_end_idx)
    data_size_in_bytes = os.path.getsize(file_path)
    real_iter_num = data_size_in_bytes / (channel_num * 4 * category_num)
    assert (np.floor(real_iter_num) == real_iter_num)
    time_end_idx = int(min(real_iter_num, time_end_idx))
    all_prob_data = np.zeros((0, (channel_end - channel_start)), dtype=np.float32)
    for cur_time_start_idx in range(time_start_idx, time_end_idx, chunk_height):
        offset = cur_time_start_idx * channel_num * 4 * category_num  # np.float32 is 4 bytes
        count = min((time_end_idx - cur_time_start_idx), chunk_height) * channel_num*category_num
        prob_data = np.fromfile(file_path, np.float32, count=count, offset=offset)
        if prob_data.size == 0:
            break
        prob_data = prob_data.reshape((-1, channel_num, category_num))
        prob_data = prob_data[::, channel_start:channel_end:1, category_idx]
        all_prob_data = np.concatenate((all_prob_data, prob_data), axis=0)
        print('prob data shape: ', all_prob_data.shape)
    print('prob data read')
    return all_prob_data


def read_all_prob_data(file_path, channel_start=0, channel_end=5150, time_start_idx=0, time_end_idx=60*2000,
                       channel_num=5150, category_num=5, chunk_height=10*60*60*5):
    assert (channel_start >= 0 and channel_end <= 5150 and (channel_end - channel_start) <= channel_num)
    assert (0 <= time_start_idx <= time_end_idx)
    data_size_in_bytes = os.path.getsize(file_path)
    real_iter_num = data_size_in_bytes / (channel_num * 4 * category_num)
    assert (np.floor(real_iter_num) == real_iter_num)
    elem_num_in_data = data_size_in_bytes/4  # np.float32 is 4 bytes
    assert(round(elem_num_in_data) == elem_num_in_data)
    elem_num_in_data = round(elem_num_in_data)
    time_end_idx = int(min(real_iter_num, time_end_idx))
    all_prob_data = np.fromfile(file_path, np.float32, count=elem_num_in_data, offset=0)
    all_prob_data = all_prob_data.reshape((-1, channel_num, category_num))
    all_prob_data = all_prob_data[::, channel_start:channel_end:1, ::]
    return all_prob_data

# file_path = 'D:\\umut\\Bayraktar_kazma_Toprak_10m--2017-08-14--15-03-54.bin'
# raw_data = read_raw_data(file_path, 0, 5150, 0, 60*2000, 5150, 0, 60*2000)


# plt.figure()
# plt.imshow(raw_data)
# plt.show()

# data = np.fromfile(file_path, np.int16, count= -1, offset = 0)
# data = data.reshape((-1, 5150))

# data = np.ones((5150*100), dtype = np.float32)
# np.save(data, )


def get_num_channel_from_info(raw_file_path):
    info_file_path = raw_file_path+'.info'
    with open(info_file_path) as json_file:
        info_file = json.load(json_file)
    # fileInfo=pd.read_json(infoFile,orient='index')
    channels = info_file['selected_channel_zones']
    start_channel = channels[0][0]
    channel_num = channels[0][1]-start_channel+1
    return start_channel, channel_num
