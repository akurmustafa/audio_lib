
import os
# import IPython.display
from utils import util_audio

cur_folder_path = os.getcwd()+'\\' 
data_folder_path = cur_folder_path+'data\\'
audio_file_name = 'yaz_gazeteci_yaz.wav'
audio_file_path = data_folder_path+audio_file_name
sr, audio_data = util_audio.read_audio(audio_file_path)
audio = util_audio.Audio(audio_file_path)
sections= [util_audio.Section(11, 15), util_audio.Section(1, 10), util_audio.Section(101, 110)]
audio.mark_to_del_sections(sections)
sr, audio_clip = audio.get_section_audio(util_audio.Section(0, 7))
audio.vis_section(util_audio.Section(0, 7), 25, 5)
# IPython.display.Audio(data=audio_clip, rate=sr)
pass