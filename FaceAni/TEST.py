from keras.models import load_model
from keras import Model
# from data_loader import get_audio_feat
import numpy as np
import csv
import os
import scipy.io.wavfile as wav
from python_speech_features import logfbank, mfcc, ssc
from config import *


# model = load_model(r"E:\ls\pyworkspace\FaceAni\AU_prediction.h5")
model = load_model(r"E:\ls\pyworkspace\FaceAni\AU_classification.h5")

(rate, sig) = wav.read(r"E:\ls\pyworkspace\FaceAni\si649.wav")
fps = 25
winstep = 1.0 / fps / mfcc_win_step_per_frame / up_sample_rate
mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=winstep, numcep=13)
logfbank_feat = logfbank(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
ssc_feat = ssc(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
full_feat = np.concatenate([mfcc_feat, logfbank_feat, ssc_feat], axis=1)
au = []
feat_cut_tail = full_feat[0:(int)(full_feat.shape[0] / 4) * 4]
f = np.array(feat_cut_tail)
for i in range((int)(full_feat.shape[0]/4) - window_size - 1):
    input = f[4 * i: 4 * (i + window_size)]
    au.append(model.predict(np.reshape(input, (1, input.shape[0], input.shape[1]))))
    # input.append(f[4 * i: 4 * (i + window_size)])
# print(input)
# input = np.array(input)
# print(input.shape)
# au = model.predict(input)
# au = model(input[0])
# print(au.shape)
au = np.array(au)
print(au.shape)

def generateOutput(au, path, filename):
    iter = au.shape[0]
    os.chdir(path)
    # creat new csv format
    f = open(filename + '_classifier.csv', 'w', newline='')
    # f = open(filename + '_predict.csv', 'w', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['frame', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c',
         ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c',
         ' AU26_c', ' AU45_c'])

    for i in range(0, iter):
        csv_writer.writerow(
            # [i, au[i][0], au[i][1], au[i][2], au[i][3], au[i][4], au[i][5], au[i][6], au[i][7], au[i][8], au[i][9], au[i][10], au[i][11], au[i][12],
            #  au[i][13], au[i][14], au[i][15], au[i][16]]
            [i, au[i][0][0], au[i][0][1], au[i][0][2], au[i][0][3], au[i][0][4], au[i][0][5], au[i][0][6], au[i][0][7], au[i][0][8], au[i][0][9], au[i][0][10], au[i][0][11], au[i][0][12],
             au[i][0][13], au[i][0][14], au[i][0][15], au[i][0][16]]
        )

generateOutput(au, r"E:\ls\pyworkspace\FaceAni", 'si649')









