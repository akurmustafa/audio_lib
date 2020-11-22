
from datetime import datetime
import glob
import json
import os


def get_files(data_folder, extension):
    all_files = []
    if os.path.exists(data_folder):
        os.chdir(data_folder)
        for file in glob.glob("*" + extension):
            all_files.append(file)
    return all_files


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def remove_extension(file_path):
    return os.path.splitext(file_path)[0]


def get_parent_folder(file_path):
    parent_folder, file_name = os.path.split(file_path)
    return parent_folder


def get_file_name(file_path):
    parent_folder, file_name = os.path.split(file_path)
    return file_name


def get_num_channel_from_info(info_file_path):
    with open(info_file_path) as json_file:
        info_file = json.load(json_file)
    # fileInfo=pd.read_json(infoFile,orient='index')
    channels = info_file['selected_channel_zones']
    try:
        start_channel = channels[0]
        channel_num = channels[1]-start_channel+1
    except:
        try:
            start_channel = channels[0][0]
            channel_num = channels[0][1]-start_channel+1
        except:
            start_channel = 0
            channel_num = int(info_file['number_of_channels_in_one_sample'])
    return start_channel, channel_num


def save_to_json(cur_dict, json_path):
    with open(json_path, 'w') as fp:
        cur_dict_pretty = json.dumps(cur_dict, indent=4)
        fp.write(cur_dict_pretty)
        # json.dump(cur_dict_pretty, fp)


def read_from_json(json_path):
    with open(json_path) as json_file:
        info_file = json.load(json_file)
    return info_file


def read_from_txt(txt_path):
    with open(txt_path) as txt_file:
        txt_content = txt_file.read()
    return txt_content


def get_today():
    today_str = datetime.today().strftime('%Y_%m_%d')
    return today_str


def split_path_into_parts(folder_path):
    # assumes folder path ands with \\ or /
    while folder_path[-1] == '\\' or folder_path[-1] == '/':
        folder_path = folder_path[0:-1:1]
    folders = []
    while 1:
        folder_path, folder = os.path.split(folder_path)
        if folder != "":
            folders.append(folder)
        elif folder_path != "":
            folders.append(folder_path)
            break
    folders.reverse()  # it reverses in place, doesnt return anything
    return folders
