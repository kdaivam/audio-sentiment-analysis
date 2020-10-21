## Program to identify emotions in a call using MLP Classifier model built.
## Output : Positive, Neutral, Negative
## @Author: Kanya D
## Created On: 2020-06-05


import pyaudio
import psycopg2
import pandas as pd
import os
import wave
import csv
import pickle
from sys import byteorder
from array import array
from struct import pack
import librosa
import soundfile
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from pydub import AudioSegment
from datetime import date, timedelta

now = str(date.today())
print(now)

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == "__main__":
    # load the saved model (after training)
    output_file_name = 'emotion_analysis_'+now+'.csv'
    
    model = pickle.load(open("model/mlp_classifier_new.model", "rb"))
    file_lt = []
    file_len_lt = []
    file_emo_lt = []
    i = 0 
    for filename in glob.glob("call_files\\2020-09-06\\*.wav"):
    # record the file (start talking)
    # extract features and reshape it
        features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
        result = model.predict(features)[0]
        audio = AudioSegment.from_file(filename)
        
    # show the result !
        fpath, call_rec_file_name = os.path.split(filename)         
        print(i,'::::',call_rec_file_name, ":::time:::", audio.duration_seconds,":::emotion:::", result)
    
        i += 1
        file_lt.append(call_rec_file_name)
        file_len_lt.append(round(audio.duration_seconds,2))
        file_emo_lt.append(result)

    df = pd.DataFrame(list(zip( file_lt , file_len_lt, file_emo_lt)), 
           columns =['filename', 'duration', 'emotion'])    


    print('output file name::::', output_file_name)
    output_file_path = 'output\\{output_file_name}'.format(output_file_name=output_file_name)
    print('output file path::::', output_file_path)
    df.to_csv(output_file_path) 


