import py_midicsv as pm
import librosa
import librosa.display
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from PIL import Image
import warnings

class Ableton_Song:
    def __init__(self, midi_filepath, audio_filepath, n_mels=128, sustain=False):
        self.midi_csv_strings = pm.midi_to_csv(midi_filepath)
        self.midi_data_start_idx = None
        for i, midi_string in enumerate(self.midi_csv_strings):
            if any(x in midi_string for x in ['Control_c', 'Note_on_c', 'Note_off_c']):
                self.midi_data_start_idx = i
                break
        '''Break data into metadata, midi data, and end track data'''
        self.meta_data = self.midi_csv_strings[0:self.midi_data_start_idx]
        self.track_end = self.midi_csv_strings[-2:]
        self.midi_data = self.midi_csv_strings[self.midi_data_start_idx:-2]

        '''Initialize and populate Numpy matrices for representing MIDI actions'''
        self.song_total_midi_ticks = int(self.track_end[0].split(', ')[1])
        self.midi_note_array = np.zeros((128, self.song_total_midi_ticks))
        self.midi_sus_array = np.zeros((1, self.song_total_midi_ticks))
        self.populate_midi_note_array(apply_sus=sustain)

        '''Initialize audio waveform and mel spectrogram'''
        self.audio_waveform, self.sample_rate = librosa.load(audio_filepath, sr=None)
        self.raw_spectrogram = librosa.stft(self.audio_waveform)
        self.db_spectrogram = np.flipud(librosa.amplitude_to_db(abs(self.raw_spectrogram)))
        self.song_total_sample_ticks = self.db_spectrogram.shape[0]
        self.n_mels = n_mels
        mel_spectrogram = librosa.feature.melspectrogram(self.audio_waveform, sr=self.sample_rate, n_mels=self.n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        self.mel_spectrogram = log_mel_spectrogram



    def map_note(self, array, note_value, note_start, note_end, velocity):
        array[note_value, note_start:note_end] = velocity

    def apply_sus_to_slice(self, start_tick, end_tick, midi_note_array):
        # important to notive that end_tick is used instead of end_tick + 1
        # the end_tick is the first moment the pedal is released
        midi_slice = midi_note_array[:, start_tick: end_tick]

    def show_midi_note_array(self):
        plt.imshow(self.midi_note_array, aspect='auto')
        plt.show()

    def show_mel_spectrogram(self):
        plt.imshow(np.flipud(self.mel_spectrogram), aspect='auto')
        plt.show()

    def show_mel_and_midi(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 20))
        axs[0].imshow(np.flipud(self.mel_spectrogram), aspect='auto', interpolation='nearest')
        axs[1].imshow(self.midi_note_array, aspect='auto', interpolation='nearest')
        plt.show()

    '''MIDI NOTE ARRAY FUNCTIONS'''
    def populate_midi_note_array(self, apply_sus=False):
        # list of midi actions to numpy array and pandas DF
        midi_values = np.genfromtxt(self.midi_data, delimiter=',', dtype=None, encoding=None)
        columns = ['track', 'tick', 'control', 'channel', 'control_num', 'velocity']
        df = pd.DataFrame(midi_values)
        df.columns = columns

        # extracting just the note presses and releases
        mask = (df['control'] == ' Note_on_c') | (df['control']  == ' Note_off_c')
        df_notes = df[mask]
        df_pedals = df[~mask]

        # sort notes by the note value (control_num) and tick (when they occur)
        # when velocity > 0, the note is being pressed, velocity == 0 is note being released

        df_sorted_notes = df_notes.sort_values(['control_num', 'tick']).reset_index(drop=True)
        df_key_press = df_sorted_notes[df_sorted_notes['control'] == ' Note_on_c'].reset_index()
        df_key_release = df_sorted_notes[df_sorted_notes['control'] == ' Note_off_c'].reset_index()


        # every note press should have a proximal note release, ensure that we have the same number of
        # presses and releases, as well as a 1:1 pairing of note presses and releases for each note
        # each row in df_key_press should be matched with a corresponding row with the same index in df_key_release
        # that specifies when that note stopped being played
        assert df_key_release.shape[0] == df_key_press.shape[0]
        assert df_key_press['control_num'].equals(df_key_release['control_num'])

        # note that 'end tick' is non inclusive
        # i.e. this is the first tick when that note stopped playing
        df_note_durations = pd.DataFrame({'start_tick': df_key_press['tick'], 'end_tick': df_key_release['tick'],
                                          'control_num': df_key_press['control_num'],
                                          'velocity': df_key_press['velocity']})

        # MIDI MATRIX W/O SUSTAIN PEDAL
        for idx, row in df_note_durations.iterrows():
            self.map_note(self.midi_note_array, row['control_num'], row['start_tick'], row['end_tick'], row['velocity'])

        if apply_sus:
            '''Assigning midi pedal arrays'''
            # midi maps for pedal presses
            df_sus = df_pedals[df_pedals['control_num'] == 64].reset_index(drop=True)
            # find duration by tick length to next action
            df_sus['duration'] = np.abs(df_sus['tick'].diff(periods=-1))
            # extend last action (usually releasing sus_pedal) to end of song
            df_sus.loc[df_sus.index[-1], 'duration'] = self.song_total_midi_ticks - df_sus.loc[df_sus.index[-1], 'tick']

            # pedal actions record variations in how far the sustain pedal is pressed (i.e. velocity of 20 vs 80)
            # however, sustain pedal is binary, either on or off.  To get the duration where the pedal is pressed
            # only the presses directly after a release matter.  This pedal press extend until the next pedal release
            # finding releases
            sus_release_indexes = df_sus['velocity'] == 0
            # presses include the first row (first time pedal is pressed) and the rows directly after a pedal release
            sus_press_indexes = pd.Series([True, *sus_release_indexes])
            # since we added action, need to pop off last element
            sus_press_indexes = sus_press_indexes[:-1]

            df_sus_releases = df_sus[sus_release_indexes]
            df_sus_presses = df_sus[sus_press_indexes]

            if df_sus_presses.shape[0] != df_sus_releases.shape[0]:
                print('sustain pedal issue')
                print(df_sus_presses.shape[0])
                print(df_sus_releases.shape[0])
                print(df_sus_releases)
                print(df_sus_presses)
                print(self.song_total_midi_ticks)
            assert df_sus_presses.shape[0] == df_sus_releases.shape[0], 'assertion error sustain'
            # MIDI tick durations where sustain pedal is pressed

            for start, end in zip(df_sus_presses['tick'], df_sus_releases['tick']):
                self.apply_sus_to_slice(start, end, self.midi_note_array)

            # midi_sus_array just for visualizing when pedal is pressed
            self.midi_sus_array = np.zeros((1, self.song_total_midi_ticks))

            # midi array for sustain pedal
            for idx, row in df_sus.iterrows():
                # mapping pedal actions to midi_sus_array
                # note_value param is 0 because midi_sus_array only has one row
                self.map_note(self.midi_sus_array, 0, row['tick'], row['tick'] + int(row['duration']), row['velocity'])
            # pedal is either on or off, so assign all on values to 60
            self.midi_sus_array[self.midi_sus_array > 0] = 60

song = Ableton_Song(r'../data/testmidi1.mid', r'../data/testaudio1.wav', sustain=True)

print(song.meta_data)
print(song.track_end)
print(song.midi_data)
print(song.audio_waveform)
print(song.db_spectrogram.shape)
print(song.song_total_sample_ticks)
print(song.song_total_midi_ticks)


song.show_mel_spectrogram()
song.show_midi_note_array()
song.show_mel_and_midi()