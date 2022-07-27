import torch
from torch import nn

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class LabelSmoothing(nn.Module):
    def __init__(self, size: int, padding_idx: int, smoothing=0.0):
        """Initialize Label Smoothing criterion

        Args:
            size (int): The number of labels.  
            padding_idx (int): _description_
            smoothing (float, optional): _description_. Defaults to 0.0.
        """
        super(LabelSmoothing, self).__init__()
        # self.criterion = nn.KLDivLoss(reduction='batchmean') # TODO: https://discuss.pytorch.org/t/kldiv-loss-reduction/109131/5. "In most cases, batchmean will suffice". If anything goes wrong, confirm this.
        self.criterion = nn.KLDivLoss(reduction='sum') # TODO: https://discuss.pytorch.org/t/kldiv-loss-reduction/109131/5. "In most cases, batchmean will suffice". If anything goes wrong, confirm this.
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # print(f"input shape in criterion: {x.shape}")
        # print(f"target shape in criterion: {self.true_dist.shape}")
        return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion ):
        self.generator = generator
        self.criterion = criterion
        
    def __call__(self, x, y, norm):
        """_summary_

        Args:
            x (torch.Tensor): B x B_out x D_model.
            y (_type_): B x B
            norm (_type_): _description_

        Returns:
            Tuple[]: _description_
        """
        x = self.generator(x) # B x B_out x num_classes
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm 
        return loss.data * norm, loss 