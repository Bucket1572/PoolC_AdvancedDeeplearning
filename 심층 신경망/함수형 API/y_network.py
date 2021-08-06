import numpy as np

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

# MNIST 데이터셋 로딩
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 레이블 개수 계산
num_labels = len(np.unique(y_train))

# 원핫 벡터 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 입력 이미지 형상 재조정 및 정규화
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 신경망 파라미터
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32

# Y-Network의 왼쪽 가지
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters

# Conv2D-Dropout-MaxPooling2D 3 계층 구성
# 계층이 지날 때마다 필터 개수를 두 배로 증가시킴 (32 - 64 - 128)
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',  # 텐서 차원 음수 방지
               activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2

# Y-Network의 오른쪽 가지
right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters

# Conv2D-Dropout-MaxPooling2D 3 계층 구성
# 계층이 지날 때마다 필터 개수를 두 배로 증가시킴 (32 - 64 - 128)
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               dilation_rate=2)(y)  # 조금 더 넓게 봐서 다채로운 특징 추출
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

# 왼쪽 가지와 오른쪽 가지의 출력을 병합
y = concatenate([x, y])

# Dense 계층에 연결하기 전 특징 맵을 벡터로 변환
y = Flatten()(y)
y = Dropout(dropout)(y)

# 출력 계층
outputs = Dense(num_labels, activation='softmax')(y)

# 함수형 API에서 모델 구축
model = Model([left_inputs, right_inputs], outputs)

# 그래프를 사용해 모델 확인
plot_model(model, to_file="cnn-y-network.png", show_shapes=True)

# 계층 텍스트 설명을 사용해 모델 확인
model.summary()

# 분류 모델 손실 함수, Adam 최적화, 분류 정확도
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 입력 이미지와 레이블로 모델 훈련
model.fit([x_train, x_train],
          y_train,
          validation_data=([x_test, x_test], y_test),
          epochs=20,
          batch_size=batch_size)

# 테스트 데이터셋에서 모델 정확도 측정
score = model.evaluate([x_test, x_test], y_test, batch_size=batch_size)
print(f"\nTest accuracy: {format(100.0 * score[1], '.1f')}%")
