import scipy.io.wavfile as wav
import numpy as np
from python_speech_features import logfbank, mfcc, ssc
from config import *
import os, math, csv

def get_audio_feat(filename):
    (rate, sig) = wav.read(r'E:\ls\人脸表情动画\数据集\vidtimit\fadg0\audio' + '\\' + filename + '.wav')
    fps = 25
    winstep = 1.0 / fps / mfcc_win_step_per_frame / up_sample_rate
    mfcc_feat = mfcc(sig, samplerate=rate, winlen=0.025, winstep=winstep, numcep=13)
    logfbank_feat = logfbank(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
    ssc_feat = ssc(sig, samplerate=rate, winlen=0.025, winstep=winstep, nfilt=26)
    full_feat = np.concatenate([mfcc_feat, logfbank_feat, ssc_feat], axis=1)
    return full_feat

def get_video_label(filename):
    os.chdir(r'E:\ls\人脸表情动画\数据集\vidtimit\fadg0\processed'+ '\\' + filename)
    au = []
    with open(filename + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            au.append([float(row[' AU01_r']) / 5.0, float(row[' AU02_r']) / 5.0, float(row[' AU04_r']) / 5.0,
                       float(row[' AU05_r']) / 5.0,
                       float(row[' AU06_r']) / 5.0, float(row[' AU07_r']) / 5.0, float(row[' AU09_r']) / 5.0,
                       float(row[' AU10_r']) / 5.0,
                       float(row[' AU12_r']) / 5.0, float(row[' AU14_r']) / 5.0, float(row[' AU15_r']) / 5.0,
                       float(row[' AU17_r']) / 5.0,
                       float(row[' AU20_r']) / 5.0, float(row[' AU23_r']) / 5.0,
                       float(row[' AU25_r']) / 5.0, float(row[' AU26_r']) / 5.0, float(row[' AU45_r']) / 5.0])
    return au

def create_dataset_csv():
    loaded_data = dict()
    loaded_data['wav'] = []
    loaded_data['au'] = []

    files = ['sa1', 'sa2', 'si649', 'si1279', 'si1909', 'sx19', 'sx109', 'sx199', 'sx289', 'sx379']
    full_feat = get_audio_feat('si649')
    nFrames_represented_by_wav = math.floor(full_feat.shape[0] / mfcc_win_step_per_frame / up_sample_rate)
    mfcc_lines = full_feat[0: nFrames_represented_by_wav * mfcc_win_step_per_frame * up_sample_rate, :].reshape(
        int(nFrames_represented_by_wav * up_sample_rate),
        int(full_feat.shape[1] * mfcc_win_step_per_frame))
    print(full_feat.shape)
    print(mfcc_lines.shape)
    # for filename in files:
    #     full_feat = get_audio_feat('si649')
    #     full_au = get_video_label('si649')
    #     feat_cut_tail = full_feat[0:(int)(full_feat.shape[0] / 4) * 4]
    #     f = np.reshape(feat_cut_tail, ((int)(feat_cut_tail.shape[0] / 4), 4, 65))
    #     a = np.array(full_au)
    #     iter = f.shape[0]
    #     for i in range(0, iter):
    #         loaded_data['wav'].append(f[i])
    #         loaded_data['au'].append(a[i])
    return loaded_data

# data = create_dataset_csv()
# print(np.array(data['wav']).shape)
# print(np.array(data['au']).shape)
