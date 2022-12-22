import torch
import math
import logging
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from base_options import options
import networks
from itertools import chain
from collections import OrderedDict, namedtuple
from utils import pm, nan
from utils.distributed import all_gather_coalesced
from typing import Optional
import os


# train models
def load_train_models(state):
    if state.train_nets_type == 'unknown_init':
        model, = networks.get_networks(state, N=1)
        return [model for _ in range(state.local_n_nets)]
    elif state.train_nets_type == 'known_init':
        return networks.get_networks(state, N=state.local_n_nets)
    elif state.train_nets_type == 'loaded':
        models = networks.get_networks(state, N=state.local_n_nets)
        with state.pretend(phase='train'):  # in case test_nets_type == same_as_train
            model_dir = state.get_model_dir()
        start_idx = state.world_rank * state.local_n_nets
        for n, model in enumerate(models, start_idx):
            model_path = os.path.join(model_dir, 'net_{:04d}'.format(n))
            model.load_state_dict(torch.load(model_path, map_location=state.device))
        logging.info('Loaded checkpoints [{} ... {}) from {}'.format(
            start_idx, start_idx + state.local_n_nets, model_dir))
        return models
    else:
        raise ValueError("train_nets_type: {}".format(state.train_nets_type))
def xentropy_cost(x_target, x_pred):
    assert x_target.size() == x_pred.size(), "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_target * logged_x_pred)
    return cost_value
def cross_entropy(pred, soft_targets, **kwargs):
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
def cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    softmax: bool = False
) -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    # print(input.shape)
    if len(input.shape) == 3:
        input = torch.squeeze(input, 1)
    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    if softmax:
        target=F.softmax(target, dim=-1)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")
def task_loss(state, output, label, **kwargs):
    if state.num_classes == 2:
        label = label.to(output, non_blocking=True).view_as(output)
        #print ("LABEL")
        #print(label)
        #print ("OUTPUT")
        #print(output)
        return F.binary_cross_entropy_with_logits(output, label, **kwargs) 
    elif state.reproduction_test:
        return F.cross_entropy(output, label.long().argmax(-1), **kwargs)
    else:
        #return xentropy_cost(label, output)
        if state.textdata:
            #return F.kl_div(output, label.float(), reduction='batchmean', **kwargs)
            return cross_entropy_with_probs(output, label, reduction='mean', softmax=state.label_softmax, **kwargs)
        else:
            #return F.cross_entropy(output, label.long().argmax(-1), **kwargs)
            return cross_entropy_with_probs(output, label, reduction='mean', softmax=state.label_softmax, **kwargs)
def task_loss_eval(state, output, label, **kwargs):
    if state.num_classes == 2:
        label = label.to(output, non_blocking=True).view_as(output)
        return F.binary_cross_entropy_with_logits(output, label, **kwargs)
    else:
        label = label.to(torch.int64)
        return F.cross_entropy(output, label, **kwargs)


def final_objective_loss(state, output, label):
    if state.mode in {'distill_basic', 'distill_adapt'}:
        return task_loss_eval(state, output, label)
    elif state.mode == 'distill_attack':
        label = label.clone()
        label[label == state.attack_class] = state.target_class
        return task_loss(state, output, label)
    else:
        raise NotImplementedError('mode ({}) is not implemented'.format(state.mode))


# NB: This trains params or model inplace!!!
def train_steps_inplace(state, models, steps, params=None, callback=None):
    if isinstance(models, torch.nn.Module):
        models = [models]
    if params is None:
        params = [m.get_param() for m in models]

    for i, (data, label, lr) in enumerate(steps):
        if callback is not None:
            callback(i, params)

        data = data.detach()
        label = label.detach()
        lr = lr.detach()

        for model, w in zip(models, params):
            model.train()  # callback may change model.training so we set here
            model.distilling_flag=True
            # print(data.shape)
            output = model.forward_with_param(data, w)
            # print("output.shape", output.shape)
            #print(output[0].size())
            # print("output", output.shape)
            # print("label", label.shape)
            loss = task_loss(state, output, label)
            lr=lr.squeeze()
            #print(lr)
            #print(lr.size())
            loss.backward(lr)
            with torch.no_grad():
                w.sub_(w.grad)
                w.grad = None

    if callback is not None:
        callback(len(steps), params)

    return params

# NOTE [ Evaluation Result Format ]
#
# Result is always a 3-tuple, containing (test_step_indices, accuracies, losses):
#
# - `test_step_indices`: an int64 vector of shape [NUM_STEPS].
# - `accuracies`:
#   + for mode != 'distill_attack', a matrix of shape [NUM_STEPS, NUM_MODELS].
#   + for mode == 'distill_attack', a tensor of shape
#       [NUM_STEPS, NUM_MODELS x NUM_CLASSES + 3], where the last dimensions
#       contains
#         [overall acc, acc w.r.t. modified labels,
#          class 0 acc, class 1 acc, ...,
#          ratio of attack_class predicted as target_class]
# - `losses`: a matrix of shape [NUM_STEPS, NUM_MODELS]


# See NOTE [ Evaluation Result Format ] for output format
def evaluate_models(state, models, param_list=None, test_all=False, test_loader_iter=None):
    n_models = len(models)
    device = state.device
    num_classes = state.num_classes
    corrects = np.zeros(n_models, dtype=np.int64)
    losses = np.zeros(n_models)
    attack_mode = state.mode == 'distill_attack'
    total = np.array(0, dtype=np.int64)
    if attack_mode:
        class_total = np.zeros(num_classes + 1, dtype=np.int64)
        # per-class acc & attacked acc. wrt target
        class_corrects = np.zeros((n_models, num_classes + 1), dtype=np.int64)
    if test_all or test_loader_iter is None:  # use raw full iter for test_all
        test_loader_iter = iter(state.test_loader)
    for model in models:
        model.eval()

    with torch.no_grad():
        for i, example in enumerate(test_loader_iter):
            if state.textdata and (not state.fairness):
                data = example.text[0]
                target = example.label
            else:
                # (data, target) = example
                data, target = example[0], example[1]
            # print(data.shape)
            # print(target.shape)
            #print(data)
            #print(data.size())
            #print(target)
            #if not state.textdata: 
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if attack_mode:
                for n in range(num_classes):
                    class_total[n] += (target == n).sum().item()

            for k, model in enumerate(models):
                model.distilling_flag=False
                if param_list is None or param_list[k] is None:
                    output = model(data)
                else:
                    output = model.forward_with_param(data, param_list[k])
                # print(output.shape)

                if num_classes == 2:
                    pred = (output > 0.5).to(target.dtype).view(-1)
                    #print("DEBUG")
                    #print (output) 
                    #print(target)
                else:
                    if state.fairness:
                        # print("Pred shape")
                        pred = torch.argmax(output, axis=1)
                    else:
                        pred = output.argmax(-1)  # get the index of the max log-probability
                    #print(output.size())
                    #print(pred.size())
                
                # print(pred.shape)

                correct_list = pred == target
                
                #print(correct_list)
                losses[k] += task_loss_eval(state, output, target, reduction='sum').item()  # sum up batch loss
                if attack_mode:
                    for c in range(num_classes):
                        class_mask = target == c
                        class_corrects[k, c] += correct_list[class_mask].sum().item()
                        if c == state.attack_class:
                            class_corrects[k, -1] += (pred[class_mask] == state.target_class).sum().item()
                corrects[k] += correct_list.sum().item()

            total += output.size(0)
            if not test_all and i + 1 >= state.test_niter:
                break
        losses /= total
        if attack_mode:
            class_total[-1] = class_total[state.attack_class]
    accs = corrects / total
    if attack_mode:
        class_accs = class_corrects / class_total[None, :]
        for n in range(num_classes):
            per_class_accs = class_accs[:, n]
        accs_wrt_wrong = corrects - class_corrects[:, state.attack_class] + class_corrects[:, -1]
        accs_wrt_wrong = accs_wrt_wrong / total
        return np.hstack((accs[:, None], accs_wrt_wrong[:, None], class_accs)), losses
    else:
        return accs, losses


def fixed_width_fmt(num, width=4, align='>'):
    if math.isnan(num):
        return '{{:{}{}}}'.format(align, width).format(str(num))
    return '{{:{}0.{}f}}'.format(align, width).format(num)[:width]


def _desc_step(state, steps, i):
    if i == 0:
        return 'before steps'
    else:
        lr = steps[i - 1][-1]
        return 'step {:2d} (lr={})'.format(i, fixed_width_fmt(lr.sum().item(), 6))


# See NOTE [ Evaluation Result Format ] for output format
def format_stepwise_results(state, steps, info, res):
    accs = res[1] * 100
    losses = res[2]
    acc_mus = accs.mean(1)
    acc_stds = accs.std(1, unbiased=True)
    loss_mus = losses.mean(1)
    loss_stds = losses.std(1, unbiased=True)

    def format_into_line(*fields, align='>'):
        single_fmt = '{{:{}24}}'.format(align)
        return ' '.join(single_fmt.format(f) for f in fields)

    msgs = [format_into_line('STEP', 'ACCURACY', 'LOSS', align='^')]
    acc_fmt = '{{: >8.4f}} {}{{: >5.2f}}%'.format(pm)
    loss_fmt = '{{: >8.4f}} {}{{: >5.2f}}'.format(pm)
    tested_steps = set(res[0].tolist())
    for at_step, acc_mu, acc_std, loss_mu, loss_std in zip(res[0], acc_mus, acc_stds, loss_mus, loss_stds):
        if state.mode == 'distill_attack':
            msgs.append('-' * 74)

        desc = _desc_step(state, steps, at_step)
        loss_str = loss_fmt.format(loss_mu, loss_std)
        acc_mu = acc_mu.view(-1)  # into vector
        acc_std = acc_std.view(-1)  # into vector
        acc_str = acc_fmt.format(acc_mu[0], acc_std[0])
        msgs.append(format_into_line(desc, acc_str, loss_str))

        if state.mode == 'distill_attack':
            msgs.append(format_into_line('acc wrt modified labels', acc_fmt.format(acc_mu[1], acc_std[1])))
            for cls_idx, (mu, std) in enumerate(zip(acc_mu[2:-1], acc_std[2:-1])):
                msgs.append(format_into_line('class {:>2} acc'.format(cls_idx), acc_fmt.format(mu, std)))
            msgs.append(format_into_line('{:>2} predicted as {:>2}'.format(state.attack_class, state.target_class),
                                         acc_fmt.format(acc_mu[-1], acc_std[-1])))

    return '{} test results:\n{}'.format(info, '\n'.join(('\t' + m) for m in msgs))


def infinite_iterator(iterable):
    while True:
        yield from iter(iterable)


# See NOTE [ Evaluation Result Format ] for output format
def evaluate_steps(state, steps, prefix, details='', test_all=False, test_at_steps=None, log_results=True):
    models = state.test_models
    n_steps = len(steps)
    #print([len(steps[i][0])for i in range(len(steps))])
    #print([len(steps[i][1])for i in range(len(steps))])
    if test_at_steps is None:
        test_at_steps = [0, n_steps]
    else:
        test_at_steps = [(x if x >= 0 else n_steps + 1 + x) for x in test_at_steps]

    test_at_steps = set(test_at_steps)
    N = len(test_at_steps)

    # cache test dataloader iter
    if test_all:
        test_loader_iter = None
    else:
        test_loader_iter = infinite_iterator(state.test_loader)

    test_nets_desc = '{} {} nets'.format(len(models), state.test_nets_type)

    def _evaluate_steps(comment, reset):  # returns Tensor [STEP x MODEL]
        if len(comment) > 0:
            comment = '({})'.format(comment)
            pbar_desc = prefix + ' ' + comment
        else:
            pbar_desc = prefix

        if log_results:
            pbar = tqdm(total=N, desc=pbar_desc)

        at_steps = []
        accs = []      # STEP x MODEL (x CLASSES)
        totals = []    # STEP x MODEL (x CLASSES)
        losses = []    # STEP x MODEL

        if reset:
            params = [m.reset(state, inplace=False) for m in models]
        else:
            params = [m.get_param(clone=True) for m in models]

        def test_callback(at_step, params):
            if at_step not in test_at_steps:  # not test_all and
                return

            acc, loss = evaluate_models(state, models, params, test_all=test_all,
                                        test_loader_iter=test_loader_iter)

            at_steps.append(at_step)
            accs.append(acc)
            losses.append(loss)
            if log_results:
                pbar.update()

        params = train_steps_inplace(state, models, steps, params, callback=test_callback)
        if log_results:
            pbar.close()

        at_steps = torch.as_tensor(at_steps, device=state.device)  # STEP
        accs = torch.as_tensor(accs, device=state.device)          # STEP x MODEL (x CLASS)
        losses = torch.as_tensor(losses, device=state.device)      # STEP x MODEL
        return at_steps, accs, losses

    if log_results:
        logging.info('')
        logging.info('{} {}{}:'.format(prefix, details, ' (test ALL)' if test_all else ''))
    res = _evaluate_steps(test_nets_desc, reset=(state.test_nets_type == 'unknown_init'))

    if state.distributed:
        rcv_lsts = all_gather_coalesced(res[1:])
        res = (
            res[0],                                      # at_steps
            torch.cat([lst[0] for lst in rcv_lsts], 1),  # accs
            torch.cat([lst[1] for lst in rcv_lsts], 1),  # losses
        )

    if log_results:
        result_title = '{} {} ({})'.format(prefix, details, test_nets_desc)
        logging.info(format_stepwise_results(state, steps, result_title, res))
        logging.info('')
    return res
