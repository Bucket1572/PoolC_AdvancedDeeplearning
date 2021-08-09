from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os

# 텐서플로의 GPU Warning 끄기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 로딩할 디렉토리
dir = "../ResNet/saved_models/cifar10_resnet-v2_.100.h5"
model = load_model(dir)

# 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocessing
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 설정
batch_size = 32

score = model.evaluate(
    x_test, y_test, batch_size=batch_size
)
print(f"\nTest accuracy: {format(100.0 * score[1], '.1f')}%")