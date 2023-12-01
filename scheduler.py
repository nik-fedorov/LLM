from torch.optim.lr_scheduler import LambdaLR


class WarmupLR(LambdaLR):

    def __init__(self, optimizer, d_model, warmup_steps):

        def lr_lambda(step):
            step += 1
            return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

        super().__init__(optimizer, lr_lambda=lr_lambda)
