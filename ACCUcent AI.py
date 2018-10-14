import os
import tensorflow as tf
from tensorflow import keras
import numpy
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##Ignore Warning

data_directory = r"C:\Users\Matt\Desktop\GitHub\ACCUcent-API\Data"

#Given a wav, returns a array of amplitudes
def parse_wave(filename):
    audio_binary = tf.read_file(filename)
    desired_channels = 1
    wav_decoder = contrib_audio.decode_wav(
        audio_binary,
        desired_channels=desired_channels)
    with tf.Session() as sess:
        sample_rate, audio = sess.run([
            wav_decoder.sample_rate,
            wav_decoder.audio
        ])
    return audio

#Given a wav, returns the accent label
def getAccentLabel(filename):
    index = filename.rfind("\\")
    wav_name = filename[index:]
    file_information = str.split(',')
    return file_information[0]

#Returns an array of all wav files across directories
def getWavFiles():
    wav_files = []
    accent_folder_file_names = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]
    # print("SIZE" + accent_folder_file_names.__len__().__str__())
    for accent_folder in accent_folder_file_names:
        nation_folder_file_names = [os.path.join(accent_folder, x) for x in os.listdir(accent_folder)]
        for nation_folder in nation_folder_file_names:
            wav_files_file_names = [os.path.join(nation_folder, y) for y in os.listdir(nation_folder) if y.endswith(".wav")]
            for wav in wav_files_file_names:
                wav_files.append(wav)

    return wav_files

def getFloatEncodedWavFiles():
    encoded_wavs = []
    wav_files = getWavFiles()

    i = 0
    for wav in wav_files:
        print(i)
        encoded_wavs.append(parse_wave(wav))
        print(encoded_wavs.__len__())
        i += 1

samples = getFloatEncodedWavFiles()
print(samples)



print("FINISHED")


