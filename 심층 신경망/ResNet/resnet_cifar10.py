import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from resnet import resnet_v1, resnet_v2
import os


def lr_schedule(epoch):
    '''
    학습률 조정

    :param epoch: 학습 횟수
    :return: lr
    '''
    lr = 1e-3
    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print(f"lr: {lr}")
    return lr


# N Dict
n_dict = {20: 3, 32: 5, 44: 7, 56: 9, 110: 18}

# CIFAR 10 데이터셋 로딩
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 하이퍼파라미터
batch_size = 32
epochs = 10
num_classes = 10
n = n_dict[20]

# 모델 버전
# 최초 논문: version = 1 (ResNet v1)
# 향상된 버전: version = 2 (ResNet v2)
version = 2

# 제공된 모델 하이퍼 파라미터 n으로부터 계산된 네트워크 깊이
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

# 요약
model.summary()
plot_model(model, to_file=f"model-v{version}.png", show_shapes=True)
print(f"model-v{version}")

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = f'cifar10_resnet-v{version}_.{epochs:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# 모델 체크포인트
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# No Data Augmentation
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks)
