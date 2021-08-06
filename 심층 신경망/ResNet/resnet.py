from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def H(inputs,
      num_filters=16,
      kernel_size=3,
      strides=1,
      activation='relu',
      batch_normalization=True,
      conv_first=True):
    '''
    Conv2D - BN - ReLU

    :param inputs: 입력 특징맵
    :param num_filters: 필터 개수
    :param kernel_size: 커널 사이즈
    :param strides: 스트라이드
    :param activation: 활성화 함수
    :param batch_normalization: 배치 정규화 여부
    :return: 출력 특징맵
    '''
    conv = Conv2D(filters=num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs

    if conv_first:
        # 컨볼루션
        x = conv(x)

        # 배치 정규화
        if batch_normalization:
            x = BatchNormalization()(x)

        # ReLU
        if activation is not None:
            x = Activation(activation)(x)
    else:
        # 배치 정규화
        if batch_normalization:
            x = BatchNormalization()(x)

        # ReLU
        if activation is not None:
            x = Activation(activation)(x)

        # 컨볼루션
        x = conv(x)
    return x


def F(inputs,
      num_filters=16,
      kernel_size=3,
      strides=1,
      batch_normalization=True,
      conv_first=True):
    '''
    Conv2D - BN

    :param inputs: 입력 특징맵
    :param num_filters: 필터 개수
    :param kernel_size: 커널 사이즈
    :param strides: 스트라이드
    :param batch_normalization: 배치 정규화 여부
    :return: 출력 특징맵
    '''
    return H(inputs=inputs,
             num_filters=num_filters,
             kernel_size=kernel_size,
             activation=None,
             strides=strides,
             batch_normalization=batch_normalization,
             conv_first=conv_first)


def transition(inputs,
               num_filters=16,
               kernel_size=3,
               strides=1,
               conv_first=True):
    return H(inputs=inputs,
             num_filters=num_filters,
             kernel_size=kernel_size,
             activation=None,
             strides=strides,
             batch_normalization=False,
             conv_first=conv_first)


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 downsample_strides_coef=1):
    '''
        ReLU(F(x) + x)

        :param inputs: 입력 특징맵
        :param num_filters: 필터 개수
        :param kernel_size: 커널 사이즈
        :param strides: 스트라이드
        :param activation: 활성화 함수
        :param batch_normalization: 배치 정규화 여부
        :param downsample_strides_coef: 다운샘플링 시 strides에 곱해지는 계수, 다운샘플링이 아니면 1
        :return: 출력 특징맵
        '''
    x = inputs

    # 컨볼루션 연산이 이루어지는 특징맵
    y = H(inputs=x,
          num_filters=num_filters,
          kernel_size=kernel_size,
          strides=downsample_strides_coef * strides,
          activation=activation,
          batch_normalization=batch_normalization)
    y = F(inputs=y,
          num_filters=num_filters,
          kernel_size=kernel_size,
          strides=strides,
          batch_normalization=batch_normalization)

    if downsample_strides_coef > 1:
        # 다운샘플링; 차원이 맞지 않음. -> 전이 계층; 1 * 1 Conv2D
        x = transition(inputs=x,
                       num_filters=num_filters,
                       kernel_size=1,
                       strides=downsample_strides_coef * strides)

    # 잔차 레이어 결과
    y = add([x, y])
    y = Activation(activation)(y)
    return y


def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])')

    # 모델 정의 시작
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    # 입력 특징맵
    inputs = Input(shape=input_shape)

    # 책에 나온 모델 구조: Input -> H -> 잔차 -> 전이 -> 잔차 -> 전이 -> 잔차 -> Average Pooling -> Flatten -> Dense
    x = H(inputs)

    # 잔차 유닛 인스턴스화
    # 모델 구조 상 잔차 구조가 3번 등장함.
    for stack in range(3):
        # 잔차 블록 반복
        for res_block in range(num_res_blocks):
            # 전이
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 downsample_strides_coef=2)  # 다운샘플링
            # 전이 아니고 잔차 구조 내
            else:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters)
        # 잔차 구조 이후 필터 2배
        num_filters *= 2

    # 분류기 추가
    # v1은 마지막 숏컷 연결-ReLU 후에는 BN을 사용하지 않는다.
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    # 출력 레이어
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # 모델 인스턴스화
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n + 2 (eg 56, 110 in [b])')

    # 모델 정의
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2에서는 2 경로로 나뉘기 전에 입력에 BN-ReLU와 함께 Conv2D 수행
    x = H(inputs=inputs,
          num_filters=num_filters_in,
          conv_first=True)

    # 잔차 유닛의 스택을 인스턴스화함
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0: # 맨 처음 잔차는 컨볼루션만
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0: # 전이 레이어
                    strides = 2 # 다운샘플링

            # 병목 잔차 유닛
            y = H(inputs=x,
                  num_filters=num_filters_in,
                  kernel_size=1,
                  strides=strides,
                  activation=activation,
                  batch_normalization=batch_normalization,
                  conv_first=False)
            y = F(inputs=y,
                  num_filters=num_filters_in,
                  conv_first=False)
            y = H(inputs=y,
                  num_filters=num_filters_out,
                  kernel_size=1,
                  conv_first=False)

            if res_block == 0:
                # 변경 된 차원에 맞춤
                x = transition(inputs=x,
                               num_filters=num_filters_out,
                               kernel_size=1,
                               strides=strides)
        num_filters_in = num_filters_out

    # 상단에 분류 모델 추가
    # v2에서는 풀링 전에 BN-ReLU를 위치시킴
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    # 출력 레이어
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # 모델 인스턴스화
    model = Model(inputs=inputs, outputs=outputs)
    return model