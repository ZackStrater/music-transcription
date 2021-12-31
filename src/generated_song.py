import py_midicsv as pm
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import warnings

class Generated_Song:
    def __init__(self, song_length=10000):

        '''Break data into metadata, midi data, and end track data'''
        self.meta_data = ['0, 0, Header, 0, 1, 96\n', '1, 0, Start_track\n', '1, 0, Title_t, "1 1-Keyzone Classic\\000"\n', '1, 0, Time_signature, 4, 2, 36, 8\n', '1, 0, Time_signature, 4, 2, 36, 8\n']
        self.track_end = [f'1, {song_length}, End_track\n', '0, 0, End_of_file']
        self.midi_data = []

        '''Initialize and populate Numpy matrices for representing MIDI actions'''
        self.song_length = song_length
        self.midi_note_array = np.zeros((128, self.song_length))

        self.midi_df = pd.DataFrame(columns=['track', 'tick', 'control', 'channel', 'control_num', 'velocity'])


    def add_note(self, note_start, note_value, note_length, velocity):
        self.midi_df.loc[len(self.midi_df.index)] = [1, note_start, 'Note_on_c', 0, note_value, velocity]
        self.midi_df.loc[len(self.midi_df.index)] = [1, note_start+note_length, 'Note_off_c', 0, note_value, velocity]
        # self.populate_midi_note_array()


    def add_randomized_notes(self, num_notes=100, time='uniform', note_value='normal', note_length='uniform', velocity='normal'):
        note_starts, note_vals, note_lens, note_vels = None, None, None, None

        if time == 'uniform':
            note_starts = np.random.uniform(low=0, high=self.song_length, size=num_notes)

        if note_value == 'normal':
            note_vals = np.random.normal(loc=64, scale=21, size=num_notes)
        elif note_value == 'uniform':
            note_vals = np.random.uniform(low=0, high=127, size=num_notes)

        if note_length == 'uniform':
            note_lens = np.random.uniform(low=10, high=1000, size=num_notes)
        elif note_length == 'normal':
            note_lens = np.random.normal(loc=100, scale=300, size=num_notes)

        if velocity == 'normal':
            note_vels = np.random.normal(loc=64, scale=21, size=num_notes)


        note_starts = np.round(note_starts)
        note_vals = np.round(note_vals)
        note_lens = np.round(note_lens)
        note_vels = np.round(note_vels)

        note_vals[note_vals > 127] = 127
        note_vals[note_vals < 0] = 0
        note_lens[note_lens < 10] = 10
        note_vels[note_vels > 127] = 127
        note_vels[note_vels < 10] = 10

        midi_values = np.column_stack((note_starts, note_vals, note_lens, note_vels))
        print(midi_values)


 # note_start, note_value, note_length, velocity
        for row in midi_values:
            self.add_note(int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            print(row)

    def sort_midi_df(self):
        self.midi_df.sort_values(by='tick', inplace=True)

    def export_midi(self, path_out):

        midi_out = []

        for line in self.meta_data:
            midi_out.append(line)

        self.sort_midi_df()
        df = self.midi_df.copy()
        cols = df.columns
        df['midi_string'] = df[cols].apply(lambda row: ', '.join(row.values.astype(str)) + '\n', axis=1)

        midi_strings = df['midi_string'].tolist()
        midi_out += midi_strings
        midi_out += self.track_end

        midi_object = pm.csv_to_midi(midi_out)
        with open(path_out, 'wb') as output_file:
            midi_writer = pm.FileWriter(output_file)
            midi_writer.write(midi_object)

song = Generated_Song()
print(song.midi_df)


song.sort_midi_df()
print(song.midi_df)
song.add_randomized_notes()
song.export_midi('../data/testmidi.mid')