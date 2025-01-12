import torch

if __name__ == '__main__':
    # GPU 설정
    device_train = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 학습 디바이스
    device_loss = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device_train)  # 손실 계산 디바이스

    print(device_train)
    print(device_loss)
    