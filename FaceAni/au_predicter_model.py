from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Masking, Embedding, Input, Multiply
from keras import losses, optimizers, Model, callbacks
from data_loader import *
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt

time_step = 4  # use 4 neighboring frames
data_dim = 65  # 13-d MFCC + 26-d MFB + 26-d SSC

model = load_model(r"E:\ls\pyworkspace\FaceAni\AU_classifier.h5")

x = Input(shape=(time_step * window_size, data_dim))
# #hidden1 = LSTM(256, return_sequences=True)(x)
# hidden2 = LSTM(256)(hidden1)
hidden2 = LSTM(256)(x)
hidden3 = Dense(256)(hidden2)
# hidden4 = Dense(128)(hidden3)
# hidden5 = Dense(17, activation='sigmoid')(hidden4)
hidden5 = Dense(17)(hidden3)
# mask = normal(model(x))
mask = model(x)
out = Multiply()([hidden5, mask])

predictor = Model(x, out)
predictor.summary()
predictor.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# predictor.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
# #
data = create_dataset_csv()
wavs = np.array(data['wav'])
au = np.array(data['au'])

# early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
early_stopping=callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001,
                              patience=10, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)

history = predictor.fit(wavs, au, batch_size=256, epochs=20, verbose=1, validation_split=0.3, callbacks = [early_stopping])

# predictor.save(r"E:\ls\pyworkspace\FaceAni\AU_prediction.h5")
#
# # 绘制训练 & 验证的准确率值
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
