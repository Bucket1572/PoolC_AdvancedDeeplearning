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
transition을 따로 두고, resnet_layer는 실제 잔차 인스턴스로 구현