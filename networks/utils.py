import torch
import torchvision
import logging
import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import math
import types
from contextlib import contextmanager
from torch.optim import lr_scheduler
from six import add_metaclass
from itertools import chain


def init_weights(net, state):
    init_type, init_param = state.init, state.init_param

    if init_type == 'imagenet_pretrained':
        assert net.__class__.__name__ == 'AlexNet'
        state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
        state_dict['classifier.6.weight'] = torch.zeros_like(net.classifier[6].weight)
        state_dict['classifier.6.bias'] = torch.ones_like(net.classifier[6].bias)
        net.load_state_dict(state_dict)
        del state_dict
        return net

    def init_func(m):
        classname = m.__class__.__name__
        if classname=='TransformerDecoder':
            for i in range(len(m.layers)):
                m.layers[i].self_attn._reset_parameters()
                m.layers[i].multihead_attn._reset_parameters()
        if classname.startswith('RNN') or classname.startswith('LSTM'):
            for names in m._all_weights:
                for name in filter(lambda n: "bias" in n,  names):
                    bias = getattr(m, name)
                    init.constant_(bias, 0.0)
                for name in filter(lambda n: "weight" in n,  names):
                    weight = getattr(m, name)
                    if init_type == 'normal':
                        init.normal_(weight, 0.0, init_param)
                    elif init_type == 'xavier':
                        init.xavier_normal_(weight, gain=init_param)
                    elif init_type == 'xavier_unif':
                        init.xavier_uniform_(weight, gain=init_param)
                    elif init_type == 'kaiming':
                        init.kaiming_normal_(weight, a=init_param, mode='fan_in')
                    elif init_type == 'kaiming_out':
                        init.kaiming_normal_(weight, a=init_param, mode='fan_out')
                    elif init_type == 'orthogonal':
                        init.orthogonal_(weight, gain=init_param)
                    elif init_type == 'zero':
                        init.zeros_(weight)
                    elif init_type == 'one':
                        init.ones(weight)
                    elif init_type == 'constant':
                        init.constant_(weight, init_param)
                    elif init_type == 'default':
                        if hasattr(weight, 'reset_parameters'):
                            weight.reset_parameters()
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
        if classname.startswith('Conv') or classname == 'Linear':
            if getattr(m, 'bias', None) is not None:
                    init.constant_(m.bias, 0.0)
            if getattr(m, 'weight', None) is not None:
                if init_type == 'normal':
                    init.normal_(m.weight, 0.0, init_param)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=init_param)
                elif init_type == 'xavier_unif':
                    init.xavier_uniform_(m.weight, gain=init_param)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
                elif init_type == 'kaiming_out':
                    init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=init_param)
                elif init_type == 'zero':
                    init.zeros_(m.weight)
                elif init_type == 'one':
                    init.ones(m.weight)
                elif init_type == 'constant':
                    init.constant_(m.weight, init_param)
                elif init_type == 'default':
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif 'Norm' in classname:
            if getattr(m, 'weight', None) is not None:
                m.weight.data.fill_(1)
            if getattr(m, 'bias', None) is not None:
                m.bias.data.zero_()
        elif classname == 'Embedding':
            m.weight.data.copy_(state.pretrained_vec)
            m.weight.requires_grad=state.learnable_embedding

    net.apply(init_func)
    return net


def print_network(net, verbose=False):
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)


def clone_tuple(tensors, requires_grad=None):
    return tuple(
        t.detach().clone().requires_grad_(t.requires_grad if requires_grad is None else requires_grad) for t in tensors)

##############################################################################
# ReparamModule
##############################################################################


class PatchModules(type):
    def __call__(cls, state, *args, **kwargs):
        r"""Called when you call ReparamModule(...) """
        net = type.__call__(cls, state, *args, **kwargs)

        # collect weight (module, name) pairs
        # flatten weights
        w_modules_names = []

        for m in net.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    w_modules_names.append((m, n))
            for n, b in m.named_buffers(recurse=False):
                if b is not None:
                    logging.warn((
                        '{} contains buffer {}. The buffer will be treated as '
                        'a constant and assumed not to change during gradient '
                        'steps. If this assumption is violated (e.g., '
                        'BatchNorm*d\'s running_mean/var), the computation will '
                        'be incorrect.').format(m.__class__.__name__, n))

        net._weights_module_names = tuple(w_modules_names)
        #print(net._weights_module_names)
        # Put to correct device before we do stuff on parameters
        net = net.to(state.device)

        ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

        assert len(set(w.dtype for w in ws)) == 1

        # reparam to a single flat parameter
        net._weights_numels = tuple(w.numel() for w in ws)
        net._weights_shapes = tuple(w.shape for w in ws)
        with torch.no_grad():
            flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

        # remove old parameters, assign the names as buffers
        for m, n in net._weights_module_names:
            delattr(m, n)
            m.register_buffer(n, None)

        # register the flat one
        net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

        return net


@add_metaclass(PatchModules)
class ReparamModule(nn.Module):
    def _apply(self, *args, **kwargs):
        rv = super(ReparamModule, self)._apply(*args, **kwargs)
        return rv

    def get_param(self, clone=False):
        if clone:
            return self.flat_w.detach().clone().requires_grad_(self.flat_w.requires_grad)
        return self.flat_w

    @contextmanager
    def unflatten_weight(self, flat_w):
        ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
        for (m, n), w in zip(self._weights_module_names, ws):
            setattr(m, n, w)
        yield
        for m, n in self._weights_module_names:
            setattr(m, n, None)

    def forward_with_param(self, inp, new_w):
        with self.unflatten_weight(new_w):
            return nn.Module.__call__(self, inp)
    
    def hidden_with_param(self, inp, new_w):
        with self.unflatten_weight(new_w):
            return self.hidden(inp)

    def __call__(self, inp):
        return self.forward_with_param(inp, self.flat_w)

    # make load_state_dict work on both
    # singleton dicts containing a flattened weight tensor and
    # full dicts containing unflattened weight tensors...
    def load_state_dict(self, state_dict, *args, **kwargs):
        if len(state_dict) == 1 and 'flat_w' in state_dict:
            return super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        with self.unflatten_weight(self.flat_w):
            flat_w = self.flat_w
            del self.flat_w
            super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
        self.register_parameter('flat_w', flat_w)

    def reset(self, state, inplace=True):
        if inplace:
            flat_w = self.flat_w
        else:
            flat_w = torch.empty_like(self.flat_w).requires_grad_()
        with torch.no_grad():
            with self.unflatten_weight(flat_w):
                init_weights(self, state)
        return flat_w
