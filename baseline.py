import os
import scipy.io.wavfile as wav
import numpy as np
import pysptk
import tqdm

"""
通过语音的基频值预测语音情绪是否激动：
基频 > 191  为 女性
基频 <= 191 为 男性

注：须安装基频提取包 pysptk，可使用命令：pip install pysptk
"""


def stat(samples, size_step=0.02, minf0=50, maxf0=500):
    neutral_mean = []
    neutral_std = []
    excited_mean = []
    excited_std = []

    for sample in tqdm.tqdm(samples):
        path = os.path.join('dataset/train/', sample[0])
        label = int(sample[1])

        fs, data_audio = wav.read(path)

        data_audio = data_audio / 2 ** 15
        size_stepS = size_step * float(fs)
        data_audiof = np.asarray(data_audio * (2 ** 15), dtype=np.float64)  # swipe: float64  rapt: float32
        F0 = pysptk.sptk.swipe(data_audiof, fs, int(size_stepS), min=minf0, max=maxf0, otype='f0')
        F0nz = F0[F0 > 0.0]
        mean = np.mean(F0nz, axis=0)
        std = np.std(F0nz, axis=0)
        if label == 0:
            neutral_mean.append(mean)
            neutral_std.append(std)
        else:
            excited_mean.append(mean)
            excited_std.append(std)

    return np.mean(np.array(neutral_mean)), np.mean(np.array(neutral_std)), np.mean(np.array(excited_mean)), np.mean(np.array(excited_std))


def predict(path, threshold_mean, threshold_std, size_step=0.02, minf0=50, maxf0=500):
    fs, data_audio = wav.read(path)

    data_audio = data_audio / 2 ** 15
    size_stepS = size_step * float(fs)
    data_audiof = np.asarray(data_audio * (2 ** 15), dtype=np.float64)  # swipe: float64  rapt: float32
    F0 = pysptk.sptk.swipe(data_audiof, fs, int(size_stepS), min=minf0, max=maxf0, otype='f0')
    F0nz = F0[F0 > 0.0]
    mean = np.mean(F0nz, axis=0)
    std = np.std(F0nz, axis=0)

    if mean > threshold_mean or std > threshold_std:  # 基频高于阈值为激动
        predicted = 1  # '激动'
    else:
        predicted = 0  # '平静'
    return predicted


if __name__ == '__main__':
    with open('dataset/train.txt', 'r') as f:
        lines = f.read().split('\n')

    samples = [(x.split('\t')[0], x.split('\t')[1]) for x in lines]

    split = int(len(samples) * 0.8)

    trainning_data = samples[:split]
    validation_data = samples[split:]

    neutral_mean, neutral_std, excited_mean, excited_std = stat(trainning_data)

    threshold_mean = (neutral_mean + excited_mean) / 2
    threshold_std = (neutral_std + excited_std) / 2

    print('threshold_mean:', threshold_mean, 'threshold_std:', threshold_std)  

    correct = 0
    neutral = []
    excited = []
    for sample in tqdm.tqdm(validation_data):
        path = os.path.join('dataset/train/', sample[0])
        predicted = predict(path, threshold_mean, threshold_std)
        groundtruth = int(sample[1])
        if predicted == groundtruth:
            correct += 1
    print('baseline accuracy:', correct/len(validation_data))  # baseline accuracy: 0.7366946778711485
