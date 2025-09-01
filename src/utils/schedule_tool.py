class PhaseTraining:
    def __init__(self, total_epochs):
        self.phase1_end = total_epochs * 0.25  # 更加注重图像重建
        self.phase2_end = total_epochs * 0.75  # 关乎对比学习部分
        # phase3: 对比微调阶段（低学习率）

    def get_loss_config(self, epoch):
        if epoch < self.phase1_end:
            return {"recon_weight": 0.5, "contrastive_weight": 1.0} # 做了个交换
        elif epoch < self.phase2_end:
            return {"recon_weight": 1.0, "contrastive_weight": 0.5}
        else:
            return {"recon_weight": 1.0, "contrastive_weight": 1.0}
