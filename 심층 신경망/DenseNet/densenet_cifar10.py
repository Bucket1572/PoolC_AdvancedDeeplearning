from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Activation, Dense
from tensorflow.keras.layers import concatenate, Dropout, AveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf
import numpy as np
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

# Warning 끄기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

input_shape = x_train.shape[1:]

# 데이터 전처리
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 상수
batch_size = 32
epochs = 10

num_classes = 10
num_dense_blocks = 3

k = 12
k_0 = 2 * k
depth = 100
num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)

compression_factor = 0.5

def lr_schedule(epoch):
    """
    학습률 갱신

    :param epoch: 학습 횟수
    :return: 학습 횟수에 따른 학습률
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# 모델 정의
inputs = Input(shape=input_shape)
x = BatchNormalization()(inputs)
x = Activation('relu')(x)
x = Conv2D(
    k_0,
    kernel_size=3,
    padding='same',
    kernel_initializer='he_normal'
)(x)
x = concatenate([inputs, x])

# 전이층으로 연결된 밀집 블록의 스택
for i in range(num_dense_blocks):
    # 밀집 블록은 병목 계층의 스택임
    for j in range(num_bottleneck_layers):
        # 병목 계층
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(
            4 * k,
            kernel_size=1,
            padding='same',
            kernel_initializer='he_normal'
        )(y)

        # 드롭아웃
        y = Dropout(0.2)(y)

        # H(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(
            k,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal'
        )(y)

        # 드롭아웃
        y = Dropout(0.2)(y)

        x = concatenate([x, y])

    # 마지막 밀집 블록 이후에는 전이 계층이 없음
    if i == num_dense_blocks - 1:
        continue

    # 전이 계층
    k_0 += num_bottleneck_layers * k
    k_0 = int(k_0 * compression_factor)
    y = BatchNormalization()(x)
    y = Conv2D(
        k_0,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal'
    )(y)

    # 드롭아웃
    y = Dropout(0.2)(y)

    # 풀링
    x = AveragePooling2D()(y)

# 분류 계층
x = AveragePooling2D(pool_size=8)(x)
y = Flatten()(x)
outputs = Dense(
    num_classes,
    kernel_initializer='he_normal',
    activation='softmax'
)(y)

# 모델을 인스턴스화하고 컴파일함.
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(1e-3),
    metrics=['accuracy']
)

model.summary()
plot_model(model, to_file=f"model-dense_net.png", show_shapes=True)

# 체크포인트를 위한 디렉토리 생성
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = f'cifar10_dense_net_.{epochs:03d}.h5'
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

model.fit(
    x = x_train,
    y = y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    shuffle=True,
    callbacks=callbacks
)
