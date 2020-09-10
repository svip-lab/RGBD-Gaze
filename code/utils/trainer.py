import os
from contextlib import contextmanager
import torch
import torch.nn
import logging
import logging.handlers
import time
from .edict import edict


class Trainer(object):
    def __init__(self, checkpoint_dir='./', is_cuda=True):
        self.checkpoint_dir = checkpoint_dir
        self.is_cuda = is_cuda
        self.temps = edict()
        self.extras = edict()
        self.meters = edict()
        self.models = edict()
        self._logger = logging.getLogger(self.__class__.__name__)
        # self.stream_handler = logging.StreamHandler(sys.stdout)
        # self.stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # self._logger.addHandler(self.stream_handler)
        self.logger = self._logger
        self.time = 0
        self._time = time.time()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def _group_weight(module, lr):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm) or isinstance(m, torch.nn.GroupNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)

        assert len(list(
            module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [
            dict(params=group_decay, lr=lr),
            dict(params=group_no_decay, lr=lr, weight_decay=0.)
        ]
        return groups

    def save_state_dict(self, filename):
        state_dict = edict()
        if not os.path.exists(self.checkpoint_dir):
            try:
                os.makedirs(self.checkpoint_dir)
            except OSError as exc:  # Guard against race condition
                raise exc
        # save models
        state_dict.models = edict()
        for name, model in self.models.items():
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            state_dict.models[name] = model.state_dict()
        # save meters
        state_dict.meters = edict()
        for name, meter in self.meters.items():
            state_dict.meters[name] = meter
        # save extras
        state_dict.extras = edict()
        for name, extra in self.extras.items():
            state_dict.extras[name] = extra

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state_dict, path, pickle_protocol=4)
        return self

    def load_state_dict(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        saved_dict = edict(torch.load(path, map_location=lambda storage, loc: storage))
        # load models
        for name, model in saved_dict.models.items():
            assert isinstance(self.models.get(name), torch.nn.Module)
            self.models[name] = self.models[name].cpu()
            self.models[name].load_state_dict(model)
        # load meters
        for name, meter in saved_dict.meters.items():
            self.meters[name] = meter
        # load extras
        for name, extra in saved_dict.extras.items():
            self.extras[name] = extra
        return self

    def _get_logger(self, name, propagate=False):
        child_logger = self._logger.getChild(name)
        child_logger.propagate = propagate
        return self._logger.getChild(name)

    def _timeit(self):
        self.time = time.time() - self._time
        self._time = time.time()
        return self.time

    @contextmanager
    def _freeze(self, model):
        cache = []
        for param in model.parameters():
            cache.append(param.requires_grad)
            param.requires_grad = False
        yield
        for param in model.parameters():
            param.requires_grad = cache.pop(0)

    def add_logger_handler(self, handler):
        self._logger.addHandler(handler)
        return self

    def timeit(self):
        self.time = time.time() - self._time
        self._time = time.time()
        return self

    def print(self, attr: str=None):
        if attr is None:
            print(self, flush=True)
            return self
        attrs = attr.split('.')
        obj = self
        for attr in attrs:
            obj = getattr(obj, attr)
        print(obj, flush=True)
        return self

    def chain_op(self, **kwargs):
        for op, args in kwargs:
            f = getattr(self, op, default=None)
            if f is None:
                raise ValueError('unrecognized op "%s"' % op)
            f(**args)
        return self

    def end(self):
        pass


class CLSTrainer(Trainer):
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred).type_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0).tolist()[0]
            res.append(correct_k / batch_size)
        return res
