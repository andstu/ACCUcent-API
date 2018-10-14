import os
import tensorflow as tf
from tensorflow import keras
import numpy
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##Ignore Warning

data_directory = os.getcwd() + '\Data'
label_reference = []

#initializes the label reference
def initizalizeLabelReference():
    accent_folder_file_names = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]
    for accent_folder in accent_folder_file_names:
        index = accent_folder.rfind("\\")
        accent = accent_folder[index + 1:]
        label_reference.append(accent)

def getIndexFromLabel(label):
    i = 0
    for x in label_reference:
        if(x == label):
            return i
        i += 1

    print("ERROR: INDEX NOT FOUND")

def getLabelFromIndex(index):
    return label_reference[index]

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
    wav_name = filename[index + 1:]
    file_information = wav_name.split('-')
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

#TODO: RETURN VALUES
def getFloatEncodedWavFiles():
    encoded_wavs = []
    #wav_labels = []
    wav_labels_shortened = [[0,0]]

    wav_files = getWavFiles()


    checkingAccentIndex = 0
    i = 0
    for wav in wav_files:

        accentIndex = getIndexFromLabel(getAccentLabel(wav))
        if(checkingAccentIndex != accentIndex):
            wav_labels_shortened.append([i,-1])
            checkingAccentIndex = accentIndex

        encoded_wavs.append(parse_wave(wav))
        #wav_labels.append(accentIndex)
        wav_labels_shortened[checkingAccentIndex][1] = i

        #THE DEBUGGING ZONE
        print("--------------------------------------------")
        print("Index: " + i.__str__())
        #print("Label: " + getLabelFromIndex(wav_labels[i]))
        #print("Label Index: " + wav_labels[i].__str__())
        print("Span: (" + wav_labels_shortened[checkingAccentIndex][
            0].__str__() + "," + wav_labels_shortened[checkingAccentIndex][1].__str__() + ")")

        i += 1
    return (wav_labels_shortened, encoded_wavs)

#TODO: RETURN DATA
def splitData(encoded_wavs, wav_labels_shortened, percentageTrained):
    for i,span in enumerate(wav_labels_shortened):
        lIndex = span[0]
        rIndex = span[1]
        size = (rIndex - lIndex) + 1
        numToTrain = percentageTrained * size

        training_set = []
        testing_set = []

        for i in range(lIndex,rIndex + 1):
            if(i < lIndex + size):
                training_set.append(encoded_wavs[i])
            else:
                testing_set.append(encoded_wavs[i])
    return (training_set, testing_set)


initizalizeLabelReference()
(lables, wavs) = getFloatEncodedWavFiles()
(train, test) = splitData(wavs, lables, .8)
print(wavs)
print(len(train) + len(test))
# samples = getFloatEncodedWavFiles()
# print(samples)

#getFloatEncodedWavFiles()