from operator import sub
import torch
from torch import nn
from typing import List
from defaults import *
from datasets import Container
import torch
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from itertools import product


def evaluate_labeled(models: List[nn.Module], data, device='cpu'):
    """Evaluate accuracy of models on source test dataset"""
    # depricated

    counts = torch.zeros([2], device=device) # [number datapoints labeled 0, num labeled 1]
    corrects_by_label = 0. # will become tensor of shape [n_heads, 2]
    with torch.no_grad():
        for batch, labels, _ in data:
            batch, labels = batch.to(device), labels.to(device)
            if len(models) > 1:
                outs = [torch.reshape(net(batch), (-1,)) for net in models]
            else:
                outs = models[0](batch)
            n_heads = len(outs)
            outs, labels = outs.round().bool(), labels.bool()
            counts += torch.stack([(~labels).sum(), labels.sum()]) # [number of 0 labels, number of 1 labels]

            corrects = (outs == labels) # true when the prediction is correct; shape [n_heads, batchsize]
            corrects_by_label += torch.stack((corrects[:,~labels].sum(-1), corrects[:,labels].sum(-1)), dim=-1) # shape [n_heads, 2 (n_classes)]
        return corrects_by_label / counts


def name(features, head):
    return 'set_' + '_'.join(str(f) for f in features.tolist()) + '_head_' + str(head)


def evaluate_unlabeled(models: List[nn.Module], data, n_features, device='cpu', get_subset_predictions=False):
    """Evaluate accuracy of models on both labels of target dataset"""

    if len(models) > 1:
        n = len(models)
    else:
        if device == 'cuda':
            n = models[0].module.get_num_heads()
        else:
            n = models[0].get_num_heads()

    performances = torch.zeros(n_features, n, device=device)

    if get_subset_predictions:
        idx_sets = torch.cartesian_prod(*[torch.LongTensor([0,1]).to(device) for _ in range(n_features)])
        subset_sizes = {tuple(idx.tolist()): 0 for idx in idx_sets}
        subset_predictions = {name(idx, h): 0 for idx in idx_sets for h in range(n)}

    total_size = 0
    count = 0
    n_diag = 0
    diag_corrects = torch.zeros(n, device=device)

    with torch.no_grad():
        for data_batch, _, targets in data:
            count+=1
            all_targets = targets.to(device)

            if len(models) > 1:
                preds = torch.stack([model(model.preprocess(data_batch, device=device)).round() for model in models], dim=0)
            else:
                preds = models[0](models[0].preprocess(data_batch, device=device)).round()

            is_diag = (all_targets == all_targets[:,0,None]).all(dim=-1)
            diag_preds  = preds[:,is_diag]
            diag_labels = all_targets[is_diag,0]
            diag_corrects += (diag_preds == diag_labels).sum(dim=-1)
            n_diag += len(diag_labels)

            if get_subset_predictions:
                for idx in idx_sets:
                    subset_size = 0
                    for target in all_targets:
                        if (target == idx).sum() == n_features: #checks if target and idx are equal at every value
                            subset_size += 1
                    subset_sizes[tuple(idx.tolist())] += subset_size
                    for h in range(n):
                        subset_pred = 0
                        for j, target in enumerate(all_targets):
                            if (target == idx).sum() == n_features and preds[h][j]==1:
                                subset_pred += 1
                        subset_predictions[name(idx, h)] += subset_pred

            for feature, target in enumerate(torch.t(all_targets)):
                for head, pred in enumerate(preds):
                    performances[feature][head] += (
                            target == pred).float().sum()  # checks how good the head "head" is on feature "feature"

            total_size += len(data_batch)

        diag_accuracy = diag_corrects / n_diag

        if get_subset_predictions:
            for idx in idx_sets:
                subset_size = subset_sizes[tuple(idx.tolist())]
                if subset_size == 0:
                    for h in range(n):
                        subset_predictions[name(idx, h)] = -1
                else:
                    for h in range(n):
                        subset_predictions[name(idx,h)] /= subset_size # gets the average of the prediction on a subset


        performances /= total_size  # gives the mean performances
        feature_performances = torch.tensor([0. for i in range(n_features)])  # performance on each feature
        for k in range(min(n_features, n)):
            best = torch.argmax(performances)
            idx = np.unravel_index(best.to('cpu'), performances.shape)
            bestfeature = idx[0]
            besthead = idx[1]
            feature_performances[bestfeature] = torch.max(performances).item()
            performances[bestfeature] = -1
            performances[:, besthead] = -1

    if get_subset_predictions:
        return diag_accuracy, feature_performances, subset_predictions
    else:
        return diag_accuracy, feature_performances


########################################## Here be saliency mapping

def evaluate_saliency(mode, model, data, wandb_log, device='cpu',
                      k_size=50, step_size=25):
    print('Generating saliency maps...')
    saliency_mapper = eval(mode)
    data_batch, _, _ = iter(data).next()
    smaps = []

    outs = model(data_batch[:num_wb_images].to(device))

    for d in data_batch[:num_wb_images]:
        smaps.append(saliency_mapper(model.to(device), d.unsqueeze(0).to(device), k_size=k_size, step_size=step_size,
                                     perturbation_type='fade',
                                     num_classes=model.get_num_heads(), device=device))
    return wandb_log


def gkern(klen, nsig):
    inp = np.zeros((klen, klen))
    inp[klen // 2, klen // 2] = 1
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


def blur(x, klen=11, ksig=5, device='cpu'):
    kern = gkern(klen, ksig)
    return F.conv2d(x, kern.to(device), padding=klen // 2)


def normalise(x):
    return (x - x.min()) / max(x.max() - x.min(), 0.0001)


def flat_perturbation(model, input, k_size=1, step_size=-1, num_classes=2, perturbation_type='fade', device='cpu'):
    with torch.no_grad():
        _, channels, input_y_dim, input_x_dim = input.shape
        output = torch.cat(model(input)).reshape(-1)
        if step_size == -1:
            step_size = k_size
        x_steps = range(0, input_x_dim - k_size + 1, step_size)
        y_steps = range(0, input_y_dim - k_size + 1, step_size)
        heatmap = torch.zeros((num_classes, len(y_steps), len(x_steps))).to(device) + 0.5
        num_occs = 0
        if perturbation_type == 'blur':
            blur_substrate = blur(input, device=device)
        hx = 0
        for x in x_steps:
            print('{}/{}'.format(x, input_x_dim - k_size + 1))
            hy = 0
            for y in y_steps:
                occ_im = input.clone()

                if perturbation_type == 'mean':
                    occ_im[:, :, y: y + k_size, x: x + k_size] = torch.mean(input[:, :, y: y + k_size, x: x + k_size],
                                                                            axis=(-1, -2), keepdims=True)
                if perturbation_type == 'fade':
                    occ_im[:, :, y: y + k_size, x: x + k_size] = 0.0
                if perturbation_type == 'blur':
                    occ_im[:, :, y: y + k_size, x: x + k_size] = blur_substrate[:, :, y: y + k_size, x: x + k_size]

                diff = (output - torch.cat(model(occ_im)).reshape(-1)).reshape(num_classes, 1, 1)
                heatmap[:, hy:hy + 1, hx:hx + 1] += diff
                num_occs += 1
                hy += 1
            hx += 1

    return normalise(F.interpolate(heatmap.unsqueeze(0), input.shape[2:])[0])


def hierarchical_perturbation(model, input, interp_mode='nearest', resize=None, perturbation_type='fade',
                              threshold_mode='mid-range', diff_func=torch.relu, max_depth=3,
                              verbose=True, **kwargs):
    if verbose: print('\nBelieve the HiPe!')
    with torch.no_grad():
        dev = input.device
        print('Using device: {}'.format(dev))
        bn, channels, input_y_dim, input_x_dim = input.shape
        dim = min(input_x_dim, input_y_dim)
        total_masks = 0
        depth = 0
        num_cells = int(max(np.ceil(np.log2(dim)), 1) / 2)
        base_max_depth = int(np.log2(dim / num_cells))
        if max_depth == -1 or max_depth > base_max_depth:
            max_depth = base_max_depth
        if verbose: print('Max depth: {}'.format(max_depth))

        def identity(x):
            return x

        if diff_func == None:
            diff_func = identity

        thresholds_d_list = []
        masks_d_list = []

        output = torch.cat(model(input)).reshape(-1)
        num_classes = model.get_num_heads()

        saliency = torch.zeros((1, num_classes, input_y_dim, input_x_dim), device=dev)
        if perturbation_type == 'blur':
            pre_b_image = blur(input.clone().cpu()).to(dev)

        while depth <= max_depth:
            masks_list = []
            b_list = []
            num_cells *= 2
            depth += 1
            if threshold_mode == 'var':
                threshold = torch.amin(saliency, dim=(-1, -2)) + (
                        (torch.amax(saliency, dim=(-1, -2)) - torch.amin(saliency, dim=(-1, -2))) / 2)
                threshold = -torch.var(threshold)
            elif threshold_mode == 'mean':
                threshold = torch.mean(saliency)
            else:
                threshold = torch.min(saliency) + ((torch.max(saliency) - torch.min(saliency)) / 2)

            print('Threshold: {}'.format(threshold))
            thresholds_d_list.append(diff_func(threshold))

            y_ixs = range(-1, num_cells)
            x_ixs = range(-1, num_cells)
            x_cell_dim = input_x_dim // num_cells
            y_cell_dim = input_y_dim // num_cells

            if (x_cell_dim == 0) or (y_cell_dim == 0) and verbose:
                print('Max Depth Reached: {}, using single element cell.'.format(depth))

            elif verbose:
                print('Depth: {}, {} x {} Cell Dim'.format(depth, y_cell_dim, x_cell_dim))
            possible_masks = 0

            for x in x_ixs:
                for y in y_ixs:
                    possible_masks += 1
                    x1, y1 = max(0, x), max(0, y)
                    if depth == max_depth:
                        x2, y2 = min(x + 1, num_cells), min(y + 1, num_cells)
                    else:
                        x2, y2 = min(x + 2, num_cells), min(y + 2, num_cells)

                    mask = torch.zeros((1, 1, num_cells, num_cells), device=dev)
                    mask[:, :, y1:y2, x1:x2] = 1.0
                    local_saliency = F.interpolate(mask, (input_y_dim, input_x_dim), mode=interp_mode) * saliency

                    if depth > 1:
                        if threshold_mode == 'var':
                            local_saliency = -torch.var(torch.amax(local_saliency, dim=(-1, -2)))
                        else:
                            local_saliency = torch.max(diff_func(local_saliency))
                    else:
                        local_saliency = 0

                    # If salience of region is greater than the average, generate higher resolution mask
                    if local_saliency >= threshold:
                        masks_list.append(abs(mask - 1))

                        if perturbation_type == 'blur':
                            b_image = input.clone()
                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim,
                            x1 * x_cell_dim:x2 * x_cell_dim] = pre_b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim,
                                                               x1 * x_cell_dim:x2 * x_cell_dim]
                            b_list.append(b_image)

                        if perturbation_type == 'mean':
                            b_image = input.clone()
                            mean = torch.mean(
                                b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim],
                                axis=(-1, -2), keepdims=True)

                            b_image[:, :, y1 * y_cell_dim:y2 * y_cell_dim, x1 * x_cell_dim:x2 * x_cell_dim] = mean
                            b_list.append(b_image)

            num_masks = len(masks_list)
            if verbose: print('Selected {}/{} masks at depth {}'.format(num_masks, possible_masks, depth))
            if num_masks == 0:
                depth -= 1
                break
            total_masks += num_masks
            masks_d_list.append(num_masks)

            while len(masks_list) > 0:
                if perturbation_type != 'fade':
                    b_imgs = b_list.pop()
                masks = masks_list.pop()

                # resize low-res masks to input size
                masks = F.interpolate(masks, (input_y_dim, input_x_dim), mode=interp_mode)

                if perturbation_type == 'fade':

                    perturbed_outputs = diff_func(output - torch.cat(model(input * masks)).reshape(-1)[0])

                else:
                    perturbed_outputs = diff_func(output - torch.cat(model(b_imgs)).reshape(-1)[0])

                if len(list(perturbed_outputs.shape)) == 1:
                    sal = perturbed_outputs.reshape(1, -1, 1, 1) * torch.abs(masks - 1)
                else:
                    sal = perturbed_outputs.reshape(1, num_classes, 1, 1) * torch.abs(masks - 1)

                saliency += sal

        if verbose: print('Used {} masks in total.'.format(total_masks))
        if resize is not None:
            saliency = F.interpolate(saliency, (resize[1], resize[0]), mode=interp_mode)
        return saliency[0]
