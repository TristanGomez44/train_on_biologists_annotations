import torch
from torch.nn import functional as F

class Loss(torch.nn.Module):
    def __init__(self,args,reduction="mean"):
        super(Loss, self).__init__()
        self.reduction = reduction
        self.args= args
    def forward(self,output,target,resDict):
        return computeLoss(self.args,output, target, resDict,reduction=self.reduction).unsqueeze(0)

def computeLoss(args, output, target, resDict,reduction="mean"):
    if args.master_net and ("master_net_pred" in resDict):
        kl = F.kl_div(F.log_softmax(output/args.kl_temp, dim=1),F.softmax(resDict["master_net_pred"]/args.kl_temp, dim=1),reduction="batchmean")
        ce = F.cross_entropy(output, target)
        loss = args.nll_weight*(kl*args.kl_interp*args.kl_temp*args.kl_temp+ce*(1-args.kl_interp))
    else:
        loss = args.nll_weight * F.cross_entropy(output, target,reduction=reduction)

    if args.sal_metr_mask_weight > 0 and "feat_pooled_masked" in resDict:
        all_feat = torch.cat((resDict["feat_pooled"],resDict["feat_pooled_masked"]),dim=0)
        nce_loss = args.sal_metr_mask_weight * info_nce_loss(all_feat)

    return loss

#From https://github.com/sthalles/SimCLR/
def info_nce_loss(features,n_views=2,temperature=0.07):
    batch_size = features.shape[0]//n_views

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     n_views * batch_size, n_views * batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return F.cross_entropy(logits, labels)
