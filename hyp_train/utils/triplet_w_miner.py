import torch
from pytorch_metric_learning import losses, miners

class TripletLossWithMiner(torch.nn.Module):
    def __init__(self, margin=1.0, scale=0.0, type_of_triplets="semihard"):
        super().__init__()
        self.margin = float(margin)
        self.scale = float(scale)
        self.loss = losses.TripletMarginLoss(margin=margin)
        use_miner = type_of_triplets is not None
        self.miner = None
        if use_miner:
            self.miner = miners.TripletMarginMiner(
                margin=margin, type_of_triplets=type_of_triplets
            )
   
    def forward(self, embeddings, labels):
        
        if hasattr(self, 'miner'):
            hard_pairs = self.miner(embeddings, labels)
            loss = self.loss(embeddings, labels, hard_pairs)
        else:
            loss = self.loss(embeddings, labels)
        return loss