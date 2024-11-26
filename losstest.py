class LidarUpsampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_points, gt_points):
        max_len = min(len(pred_points["feat"]), len(gt_points["feat"]))
        pred = pred_points["feat"][:max_len]
        gt = gt_points["coord"][:max_len]
        loss = self.criterion(pred, gt)
        return loss

