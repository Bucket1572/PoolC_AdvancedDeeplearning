ResNet Implement
===
---
개요
---
책에 나와있는 모델과는 조금 다르게 구현했습니다.

---
특징
---
1. 책에서는 Data Augmentation을 선택할 수 있었으나 여기서는 불가능
2. 책에서는 H, F가 없고, resnet_layer로 통합했지만, 여기서는 H, F,
transition을 따로 두고, resnet_layer는 실제 잔차 인스턴스로 구현함. 
   → 다만, resnet v2는 공통 사항 묶기가 쉽지 않아서 책과 같이 함.
3. Data Augmentation이나 평균 제거와 같은 기능은 구현하지 않음. 단순 ResNet만
4. val_acc가 작동하지 않아 val_accuracy로 monitor값을 바꿈.

---
resnet
---
ResNet을 Implement한 함수들이 있는 클래스. 전체적인 모델 구조가 나와 있음.

---
resnet_cifar10
---
실제 CIFAR10을 이용하여 학습하는 스크립트
