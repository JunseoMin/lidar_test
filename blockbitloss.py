import torch

def xnor_loss_gpu(pred, target, block_size=2**20):
    """
    XNOR 기반 손실 함수 (GPU 메모리 효율적 구현)
    Args:
        pred: 예측 값 (torch.Tensor, dtype=torch.int64, shape [N])
        target: 타겟 값 (torch.Tensor, dtype=torch.int64, shape [N])
        block_size: 블록 단위 처리 크기 (기본값: 2^20)
    Returns:
        손실 값 (float)
    """
    # GPU에 데이터 이동
    pred = pred.cuda()
    target = target.cuda()
    
    # 총 비트 수 계산
    total_bits = pred.numel() * pred.element_size() * 8
    
    # 유사성 합계를 저장할 변수 초기화
    total_similarity = 0
    
    # 블록 단위로 처리
    num_blocks = (pred.numel() + block_size - 1) // block_size
    for i in range(num_blocks):
        # 블록 슬라이스
        start = i * block_size
        end = min((i + 1) * block_size, pred.numel())
        pred_block = pred[start:end]
        target_block = target[start:end]
        
        # XNOR 연산
        xnor_result = ~(pred_block ^ target_block)
        
        # 블록 단위로 1의 개수 계산
        similarity = torch.sum(torch.bitwise_and(xnor_result, torch.full_like(xnor_result, -1)))
        total_similarity += similarity.item()
    
    # GPU 메모리 해제
    del pred, target
    torch.cuda.empty_cache()
    
    # 손실 계산 (1 - 유사성 비율)
    loss = 1 - (total_similarity / total_bits)
    return loss

# 예제 데이터 (2^20 비트를 처리하기 위한 데이터)
num_bits = 2**24  # 테스트용 (2^24 비트)
pred = torch.randint(0, 2**64, (num_bits,), dtype=torch.int64)
target = torch.randint(0, 2**64, (num_bits,), dtype=torch.int64)

# 손실 계산
loss = xnor_loss_gpu(pred, target)
print(f"손실 값: {loss}")
