import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

# 텐서플로의 GPU Warning 끄기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MNIST 데이터셋 로딩
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 레이블 개수 계산
num_labels = len(np.unique(y_train))

# 원핫 벡터 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 입력 이미지 형상 재조정 및 정규화
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 신경망 파라미터
# 이미지는 그대로 처리 (회색조, 정사각형 이미지)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3
epochs = 100

# print(f"{x_test.shape}, {y_test.shape}")

# 함수형 API를 사용해 CNN 구축
# 입력 계층
inputs = Input(shape=input_shape)

# 피드포워딩
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)

# 밀집 계층에 연결하기 전 이미지를 벡터로 변환
y = Flatten()(y)

# 드롭아웃 정규화
y = Dropout(dropout)(y)

# 출력 계층
outputs = Dense(num_labels, activation='softmax', kernel_initializer="he_normal")(y)

# 입력/출력을 제공해 모델 구축
model = Model(inputs=inputs, outputs=outputs)

# 모델 요약
model.summary()
plot_model(model, to_file=f"model-basic_cnn.png", show_shapes=True)
print("basic cnn")

# 분류 모델 손실 함수, Adam 최적화, 정확도
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 체크포인트 시스템
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = f'cifar10_basic_cnn_.{epochs:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# 모델 체크포인트
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

# 입력 이미지와 레이블로 모델 훈련
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[checkpoint])

# 테스트 데이터셋에 대한 모델 정확도
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"\nTest accuracy: {format(100.0 * score[1], '.1f')}%")
