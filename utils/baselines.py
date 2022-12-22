import torch
import logging
import numpy as np
import torch.nn as nn


def encode(d, state):
    encoder = nn.Embedding(state.ntoken, state.ninp).to(state.device)
    encoder.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
    encoder.weight.requires_grad = False
    out=encoder(d)
    out.unsqueeze_(1)
    return out
def get_baseline_label_for_one_step(state):
    if state.num_classes==2:
        dl_array = [[i==j for i in range(1)]for j in state.init_labels]*state.distilled_images_per_class_per_step
    else:
        dl_array = [[i==j for i in range(state.num_classes)]for j in state.init_labels]*state.distilled_images_per_class_per_step
    label=torch.tensor(dl_array,dtype=torch.long, requires_grad=False, device=state.device)
    if state.mode == 'distill_attack': #THIS MAY BE BROKEN NOW
        label[state.attack_class] = state.target_class
    #print(label)
    return label
    '''
    label = torch.tensor(list(range(state.num_classes)), device=state.device)
    if state.mode == 'distill_attack':
        label[state.attack_class] = state.target_class
    label = label.repeat(state.distilled_images_per_class_per_step, 1)  # [[0, 1, 2, ...], [0, 1, 2, ...], ...]
    return label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]'''
    


def random_train(state):
    data_list = [[] for _ in range(state.num_classes)]
    needed = state.distill_steps * state.distilled_images_per_class_per_step
    counts = np.zeros((state.num_classes))
    for it, example in enumerate(state.train_loader):
        if state.textdata and (not state.fairness):
            datas = example.text[0]
            labels = example.label
        else:
            (datas, labels) = example
        datas=datas.to(state.device, non_blocking=True)
        if state.textdata and (not state.fairness):
            datas=encode(datas, state)
        for data, label in zip(datas, labels):
            label_id = label.item()
            if counts[label_id] < needed:
                counts[label_id] += 1
                data_list[label_id].append(data)
                if np.sum(counts) == needed * state.num_classes:
                    break
    steps = []
    label = get_baseline_label_for_one_step(state)
    #print(counts)
    for i in range(0, needed, state.distilled_images_per_class_per_step):
        temp=(cd[i:(i + state.distilled_images_per_class_per_step)] for cd in data_list)
        #print([len(cd) for cd in data_list])
        data = sum(temp, [])
        data = torch.stack(data, 0)
        #print(data.shape)
#        while 0 in counts:
#            ind=counts.find(0)
#            counts.pop(ind)
#            label.pop(ind)
        steps.append((data, label))
    return [s for _ in range(state.distill_epochs) for s in steps]

def average_train(state):
    if state.textdata:
        sum_images = torch.zeros(
        state.num_classes, state.nc, state.input_size, state.ninp,
        device=state.device, dtype=torch.double)
        if state.fairness:
            sum_images = torch.zeros(
                state.num_classes, 
                state.nc, 
                state.ninp,
                device=state.device, 
                )
    else:
        sum_images = torch.zeros(
        state.num_classes, state.nc, state.input_size, state.input_size,
        device=state.device, dtype=torch.double)
    counts = torch.zeros(state.num_classes, dtype=torch.long)
    for it, example in enumerate(state.train_loader):
        if state.textdata and (not state.fairness):
            data = example.text[0]
            label = example.label
        else:
            data, label = example[0], example[1]
        data=data.to(state.device, non_blocking=True)
        if state.textdata and (not state.fairness):
            data=encode(data, state)
        for i, (d, l) in enumerate(zip(data, label)):
            d=d.to(sum_images)
            if not state.fairness:
                sum_images[l].add_(d)
            else:
                sum_images[l].add_(d.reshape(1,-1))
            counts[l] += 1
    if not state.fairness:
        mean_imgs = sum_images / counts[:, None, None, None].to(state.device, torch.double)
        mean_imgs = mean_imgs.to(torch.float)
        mean_imgs = mean_imgs.repeat(state.distilled_images_per_class_per_step, 1, 1, 1, 1)
        mean_imgs = mean_imgs.transpose(0, 1).flatten(end_dim=1)
        label = get_baseline_label_for_one_step(state)
    else:
        mean_imgs = sum_images / counts[:, None, None].to(state.device, torch.double)
        mean_imgs = mean_imgs.to(torch.float)
        mean_imgs = mean_imgs.repeat(state.distilled_images_per_class_per_step, 1, 1, 1)
        mean_imgs = mean_imgs.transpose(0, 1).flatten(end_dim=1)
    return [(mean_imgs, label) for _ in range(state.distill_epochs) for _ in range(state.distill_steps)]


def kmeans_train(state, p=2):
    k = state.distilled_images_per_class_per_step * state.distill_steps

    if k == 1:
        return average_train(state)

    cls_data = [[] for _ in range(state.num_classes)]

    for it, example in enumerate(state.train_loader):
        if state.textdata:
            data = example.text[0]
            label = example.label
        else:
            (data, label) = example
        data=data.to(state.device, non_blocking=True)
        if state.textdata:
            data=encode(data, state)
        for d, l in zip(data, label):
            cls_data[l.item()].append(d.flatten())
    cls_data = [torch.stack(coll, 0).to(state.device) for coll in cls_data]

    # kmeans++
    cls_centers = []
    for c in range(state.num_classes):
        if state.textdata:
            c_center = torch.empty(k, state.nc * state.input_size * state.ninp, device=state.device)
        else:
            c_center = torch.empty(k, state.nc * state.input_size * state.input_size, device=state.device)
        c_data = cls_data[c]
        # first is uniform
        c_center[0] = c_data[torch.randint(len(c_data), ()).item()]
        for i in range(1, k):
            assert p == 2
            dists_sq = (c_data[:, None, :] - c_center[:i]).pow(2).sum(dim=2)  # D x I
            weights = dists_sq.min(dim=1).values
            # A-res
            r = torch.rand_like(weights).pow(1 / weights)
            c_center[i] = c_data[r.argmax().item()]
        cls_centers.append(c_center)

    cls_centers = torch.stack(cls_centers, dim=0)
    cls_assign = [torch.full((coll.size(0),), -1, dtype=torch.long, device=state.device) for coll in cls_data]

    def iterate(n=1024):
        nonlocal cls_centers
        changed = torch.tensor(0, dtype=torch.long, device=state.device)
        cls_totals = torch.zeros_like(cls_centers)
        cls_counts = cls_totals.new_zeros(state.num_classes, k, dtype=torch.long)
        for c in range(state.num_classes):
            c_center = cls_centers[c]
            c_total = cls_totals[c]
            c_count = cls_counts[c]
            for d, a in zip(cls_data[c].split(n, dim=0), cls_assign[c].split(n, dim=0)):
                new_a = torch.norm(
                    d[:, None, :] - c_center,
                    dim=2, p=p,
                ).argmin(dim=1)
                c_total.index_add_(0, new_a, d)
                c_count.index_add_(0, new_a, c_count.new_ones(d.size(0)))
                changed += (a != new_a).sum()
                a.copy_(new_a)
            # keep empty clusters unchanged
            empty = (c_count == 0)
            nempty = empty.sum().item()
            if nempty > 0:
                logging.warn("{} empty cluster(s) found for class of index {} (kept unchanged)".format(nempty, c))
                c_count[empty] = 1
                c_total[empty] = c_center[empty]
        cls_centers = cls_totals / cls_counts.unsqueeze(2).to(cls_totals)
        return changed.item()

    logging.info('Compute {}-means with {}-norm ...'.format(k, p))
    changed = 1
    i = 0
    while changed > 0:
        changed = iterate()
        i += 1
        logging.info('\tIteration {:>3d}: {:>6d} samples changed cluster label'.format(i, changed))

    logging.info('done')

    label = get_baseline_label_for_one_step(state)
    
    if state.textdata:
        per_step_imgs = cls_centers.view(
                state.num_classes, state.distill_steps, state.distilled_images_per_class_per_step, state.nc,
                state.input_size, state.ninp).transpose(0, 1).flatten(1, 2).unbind(0)

    else:
        per_step_imgs = cls_centers.view(
                state.num_classes, state.distill_steps, state.distilled_images_per_class_per_step, state.nc,
                state.input_size, state.input_size).transpose(0, 1).flatten(1, 2).unbind(0)

    return [(imgs, label) for _ in range(state.distill_epochs) for imgs in per_step_imgs]
