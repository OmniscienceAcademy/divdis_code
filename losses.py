import torch
from torch import nn
from typing import List


def get_labeled_loss(models: List[nn.Module], criterion: nn.Module, batch, labels: torch.Tensor,
                     device='cpu', log=False) -> torch.Tensor:

    if len(models) > 1:
        outs = torch.stack([torch.reshape(net(net.preprocess(batch, device=device)), (-1,)) for net in models], dim=0)
    else:
        model = models[0]
        outs = model(model.preprocess(batch, device=device))
    labels = labels.to(device)
    n_heads = outs.size(0)

    loss = criterion(outs, labels.repeat(n_heads,1).float())
    return loss


def MI_unlabeled_losses(outs: torch.Tensor, device='cpu'):
    # mutual information loss for divdis
    mi_loss = torch.Tensor([0]).to(device).sum()
    for i1, md1 in enumerate(outs):
        for i2, md2 in enumerate(outs):
            if i1 < i2:  # we don't want to compute the mutual information of a head with itself, nor repeat the calculations
                marginal_dist = torch.zeros(4).to(device)
                marginal_dist[0] = md1.mean() * md2.mean()
                marginal_dist[1] = (1 - md1.mean()) * md2.mean()
                marginal_dist[2] = md1.mean() * (1 - md2.mean())
                marginal_dist[3] = (1 - md1.mean()) * (1 - md2.mean())
                joint_dist = torch.zeros(4).to(device)
                joint_dist[0] = (md1 * md2).mean()
                joint_dist[1] = ((1 - md1) * md2).mean()
                joint_dist[2] = (md1 * (1 - md2)).mean()
                joint_dist[3] = ((1 - md1) * (1 - md2)).mean()

                mi_loss += torch.nn.KLDivLoss(reduction="batchmean")(joint_dist.log(), marginal_dist)

    return mi_loss

def KL_unlabeled_losses(outs: torch.Tensor, device='cpu'):
    # regularisation loss for divdis
    kl_loss = 0
    for md in outs:
        even_dist = torch.tensor((1 / 2., 1 / 2.)).to(device)
        total_prob_dist = torch.zeros(2).to(device)
        total_prob_dist[0] = md.mean()
        total_prob_dist[1] = 1 - md.mean()
        kl_loss += torch.nn.KLDivLoss(reduction="batchmean")(total_prob_dist.log(), even_dist)

    return kl_loss/len(outs)

def get_class_certainty_loss(criterion: nn.Module, outs: torch.Tensor, device='cpu'):
    # loss for being uncertain about the images
    # check that loss is computed element-wise

    assert criterion.reduction == 'none'

    # get batch size
    batch_size = len(outs[0])

    # get losses against 0's and 1's
    losses_against_0 = criterion(outs, torch.zeros_like(outs))
    losses_against_1 = criterion(outs, torch.ones_like(outs))
    return (losses_against_0 * losses_against_1).mean()


def get_labeled_feature_losses(criterion: nn.Module, outs: torch.Tensor, hidden_labels: torch.Tensor, device='cpu'):
    losses = [criterion(out, hidden_labels[:,i].float()) for i, out in enumerate(outs)]
    cum_loss = 0
    for out_loss in losses:
        for loss in out_loss:
            cum_loss += loss

    return cum_loss/len(losses)


def get_unlabeled_losses(models: List[nn.Module], criterion: nn.Module, batch, hidden_labels: torch.Tensor,
                         loss_type = 0, device='cpu', log=False, smooth=False, dynamic=False):
    # gets the MI and regularisation losses (for divdis)
    if len(models) > 1:
        outs = torch.stack([torch.reshape(net(net.preprocess(batch, device=device)), (-1,)) for net in models], dim=0)
    else:
        model = models[0]
        outs = model(model.preprocess(batch, device=device))
    hidden_labels = hidden_labels.to(device)

    if loss_type==1:
        return MI_unlabeled_losses(outs, device=device), KL_unlabeled_losses(outs, device=device)

    # Sanity checking for baseline – uses labels
    if loss_type==2:
        return get_labeled_feature_losses(criterion, outs, hidden_labels, device=device), get_class_certainty_loss(
            criterion, outs, device=device)




