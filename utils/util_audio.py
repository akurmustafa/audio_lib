
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

def read_audio(audio_path):
    sr, audio_data = wavfile.read(audio_path)
    return sr, audio_data

def check_section_list_valid(section_list):
    section_list.sort()
    n_section_list = len(section_list)
    for i in range(1, n_section_list):
        if section_list[i].beg<section_list[i-1].end:
            # crossing sections
            return False
    return True

class Section(object):
    def __init__(self, beg, end):
        assert (end > beg and beg >= 0), 'arguments are not valid'
        self.beg = beg
        self.end = end

    def __lt__(self, other):
        return self.beg < other.beg

    def __str__(self):
        return '['+str(self.beg)+', '+str(self.end)+'] msec'

class Audio(object):
    def __init__(self, data_path):
        self.path = data_path
        self.sr, self.data = read_audio(data_path)
        self.ch_num = self.data.shape[1]
        self.audio_length = self.data.shape[0]
        self.to_add_places = []
        self.to_del_places = []
    
    # marked section in secs
    def mark_to_add_places(self, *args):
        for cur_arg in args:
            if cur_arg*self.sr < self.audio_length:
                self.to_add_places.append(cur_arg)
            else:
                print('Input given '+str(cur_arg)+ ' is not within the record range')
                raise ValueError('Argument of the mark_to_add_places is not valid')
        self.to_add_places.sort()
    
    def mark_to_del_sections(self, sections):
        for cur_section in sections:
            if cur_section.end*self.sr < self.audio_length:
                self.to_del_places.append(cur_section)
            else:
                print('End of given section '+cur_section+ 'passes the record length')
                raise ValueError('Argument of the mark_to_del_sections member is not valid')
        self.to_del_places.sort()
        assert check_section_list_valid(self.to_del_places), 'Given Sections are Crossing'
    
    def get_section_audio(self, section):
        assert section.end*self.sr < self.audio_length, 'section is not completely within the audio'
        audio_clip = self.data[math.floor(section.beg*self.sr):math.floor(section.end*self.sr):,::]
        return self.sr, audio_clip

    def vis_section(self, section, window_ms, noverlap_ms):
        assert section.end*self.sr < self.audio_length, 'section is not completely within the audio'
        audio_clip = self.data[math.floor(section.beg*self.sr):math.floor(section.end*self.sr):,::]
        audio_clip_single_ch = np.mean(audio_clip, axis=1)
        nfft = round(window_ms*self.sr/1000)
        noverlap = round(noverlap_ms*self.sr/1000)
        f, t, Sxx = signal.spectrogram(audio_clip_single_ch, fs=self.sr, noverlap=noverlap, nfft=nfft)
        Sxx_log = 10*np.log10(Sxx+1e-6)
        plt.figure()
        plt.pcolormesh(t, f, Sxx_log, cmap=plt.get_cmap('jet'))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()
        plt.show(block=True)
