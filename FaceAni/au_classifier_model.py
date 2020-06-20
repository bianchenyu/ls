import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking, Embedding
from keras import losses, optimizers, callbacks
from data_loader import *
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt

time_step = 4  # use 4 neighboring frames
data_dim = 65  # 13-d MFCC + 26-d MFB + 26-d SSC

model = Sequential()
model.add(LSTM(256, input_shape=(time_step * window_size, data_dim)))
model.add(Dense(256))
model.add(Dense(17, activation='sigmoid'))
# rmsprop = keras.optimizers.RMSprop(lr=0.00001)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy', 'binary_accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'binary_accuracy'])
# model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.])

# # plot_model(model, to_file='model.png')

data = create_dataset_csv()
wavs = np.array(data['wav'])
au = np.array(data['au_c'])
# print(wavs)
# print(au)
#
# print(wavs.shape)
# print(au.shape)
#
#history = LossHistory()
early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                              patience=10, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)


# history = model.fit(wavs, au, batch_size=256, epochs=10, verbose=1, validation_split=0.3, callbacks=[early_stopping])
history = model.fit(wavs, au, batch_size=256, epochs=100, verbose=1, validation_split=0.3)
model.save(r"E:\ls\pyworkspace\FaceAni\AU_classification.h5")

# 绘制训练 & 验证的准确率值
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

#模型评估
# score = model.evaluate(testwavs, testau, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])





