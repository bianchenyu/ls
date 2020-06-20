import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import logfbank, mfcc, ssc
from config import *
import os, math, csv

def get_audio_feat(path, filename):
    # (rate, sig) = wav.read(r'E:\ls\人脸表情动画\数据集\vidtimit\fadg0\audio' + '\\' + filename + '.wav')
    (rate, sig) = wav.read(path + r'\audio' + '\\' + filename + '.wav')
    fps = 25
    winstep = 1.0 / fps / mfcc_win_step_per_frame / up_sample_rate
    mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=winstep, numcep=13)
    logfbank_feat = logfbank(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
    ssc_feat = ssc(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
    full_feat = np.concatenate([mfcc_feat, logfbank_feat, ssc_feat], axis=1)
    return full_feat

def get_video_label(path, filename):
    # os.chdir(r'E:\ls\人脸表情动画\数据集\vidtimit\fadg0\processed'+ '\\' + filename)
    os.chdir(path + r'\processed'+ '\\' + filename)
    au = []
    au_c = []
    with open(filename + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            au.append([row[' AU01_r'], row[' AU02_r'], row[' AU04_r'], row[' AU05_r'], row[' AU06_r'], row[' AU07_r'], row[' AU09_r'], row[' AU10_r'],
                 row[' AU12_r'], row[' AU14_r'], row[' AU15_r'], row[' AU17_r'], row[' AU20_r'], row[' AU23_r'],
                 row[' AU25_r'], row[' AU26_r'], row[' AU45_r']])
            au_c.append([row[' AU01_c'], row[' AU02_c'], row[' AU04_c'],
                 row[' AU05_c'], row[' AU06_c'], row[' AU07_c'], row[' AU09_c'], row[' AU10_c'], row[' AU12_c'],
                 row[' AU14_c'], row[' AU15_c'], row[' AU17_c'], row[' AU20_c'], row[' AU23_c'], row[' AU25_c'],
                 row[' AU26_c'], row[' AU45_c']])
            # au.append([float(row[' AU01_r']) / 5.0, float(row[' AU02_r']) / 5.0, float(row[' AU04_r']) / 5.0, float(row[' AU05_r']) / 5.0,
            #            float(row[' AU06_r']) / 5.0, float(row[' AU07_r']) / 5.0, float(row[' AU09_r']) / 5.0, float(row[' AU10_r']) / 5.0,
            #            float(row[' AU12_r']) / 5.0, float(row[' AU14_r']) / 5.0, float(row[' AU15_r']) / 5.0, float(row[' AU17_r']) / 5.0,
            #            float(row[' AU20_r']) / 5.0, float(row[' AU23_r']) / 5.0,
            #            float(row[' AU25_r']) / 5.0, float(row[' AU26_r']) / 5.0, float(row[' AU45_r']) / 5.0])
            # au.append([row[' AU06_c'], row[' AU09_c'], row[' AU10_c'], row[' AU12_c'],
            #      row[' AU14_c'], row[' AU15_c'], row[' AU17_c'], row[' AU20_c'], row[' AU23_c'], row[' AU25_c'],
            #      row[' AU26_c']])

    return au, au_c

def create_dataset_csv():
    loaded_data = dict()
    loaded_data['wav'] = []
    loaded_data['au'] = []
    loaded_data['au_c'] = []

    # files = ['sa1', 'sa2', 'si649', 'si1279', 'si1909', 'sx19', 'sx109', 'sx199', 'sx289', 'sx379']
    datapath = r"E:\ls\人脸表情动画\数据集\vidtimit"
    dataList = os.listdir(datapath)
    for data in dataList:
        path = datapath + '\\' + data
        files = os.listdir(path + '\\audio')
        for filename in files:
            full_feat = get_audio_feat(path, filename[:-4])
            full_au, full_au_c = get_video_label(path, filename[:-4])
            feat_cut_tail = full_feat[0:(int)(full_feat.shape[0] / 4) * 4]

            #create slider window
            a = np.array(full_au)
            a_c = np.array(full_au_c)
            f = np.array(feat_cut_tail)
            print(filename)
            print(f.shape)
            print(a.shape)
            # f = np.reshape(feat_cut_tail, ((int)(feat_cut_tail.shape[0] / 4), 4, 65))
            for i in  range(len(full_au) - window_size - 1):
                loaded_data['wav'].append(f[4*i: 4*(i + window_size)])
                loaded_data['au'].append(a[i + window_size])
                loaded_data['au_c'].append(a_c[i + window_size])
    # loaded_data['wav'] = np.array(loaded_data['wav'])
    # loaded_data['au'] = np.array(loaded_data['au'])
    # loaded_data['au_c'] = np.array(loaded_data['au_c'])
    return loaded_data

# data = create_dataset_csv()
# print(np.array(data['wav']).shape)
# print(np.array(data['au']).shape)
# print(np.array(data['au_c']).shape)