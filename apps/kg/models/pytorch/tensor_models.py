"""
Knowledge Graph Embedding Models.
1. TransE
2. DistMult
3. ComplEx
4. RotatE
5. pRotatE
6. TransH
7. TransR
8. TransD
9. RESCAL
"""
import os
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT

import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from _thread import start_new_thread
import traceback
from functools import wraps

from .. import *

logsigmoid = functional.logsigmoid

def get_device(args):
    return th.device('cpu') if args.gpu[0] < 0 else th.device('cuda:' + str(args.gpu[0]))

norm = lambda x, p: x.norm(p=p)**p
get_scalar = lambda x: x.detach().item()
reshape = lambda arr, x, y: arr.view(x, y)
cuda = lambda arr, gpu: arr.cuda(gpu)

def thread_wrapped_func(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

@thread_wrapped_func
def async_update(args, emb, queue):
    th.set_num_threads(8)
    while True:
        (grad_indices, grad_values, gpu_id) = queue.get()
        clr = emb.args.lr
        if grad_indices is None:
            return
        with th.no_grad():
            grad_sum = (grad_values * grad_values).mean(1)
            device = emb.state_sum.device
            if device != grad_indices.device:
                grad_indices = grad_indices.to(device)
            if device != grad_sum.device:
                grad_sum = grad_sum.to(device)

            emb.state_sum.index_add_(0, grad_indices, grad_sum)
            std = emb.state_sum[grad_indices]  # _sparse_mask
            if gpu_id >= 0:
                std = std.cuda(gpu_id)
            std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
            tmp = (-clr * grad_values / std_values)
            if tmp.device != device:
                tmp = tmp.to(device)
            emb.emb.index_add_(0, grad_indices, tmp)

class ExternalEmbedding:
    def __init__(self, args, num, dim, device):
        self.gpu = args.gpu
        self.args = args
        self.trace = []

        self.emb = th.empty(num, dim, dtype=th.float32, device=device)
        self.state_sum = self.emb.new().resize_(self.emb.size(0)).zero_()
        self.state_step = 0
        self.async_q = None
        self.async_p = None
        self.lock = mp.Lock()

    def init(self, emb_init):
        INIT.uniform_(self.emb, -emb_init, emb_init)
        INIT.zeros_(self.state_sum)

    def share_memory(self):
        self.emb.share_memory_()
        self.state_sum.share_memory_()

    def __call__(self, idx, gpu_id=-1, trace=True):
        self.lock.acquire()
        s = self.emb[idx]
        self.lock.release()
        if gpu_id >= 0:
            s = s.cuda(gpu_id)
        # During the training, we need to trace the computation.
        # In this case, we need to record the computation path and compute the gradients.
        if trace:
            data = s.clone().detach().requires_grad_(True)
            self.trace.append((idx, data))
        else:
            data = s
        return data

    def update(self, gpu_id=-1):
        self.state_step += 1
        with th.no_grad():
            for idx, data in self.trace:
                grad = data.grad.data
                clr = self.args.lr
                #clr = self.args.lr / (1 + (self.state_step - 1) * group['lr_decay'])

                # the update is non-linear so indices must be unique
                grad_indices = idx
                grad_values = grad
                if self.async_q is not None:
                    grad_indices.share_memory_()
                    grad_values.share_memory_()
                    self.async_q.put((grad_indices, grad_values, gpu_id))
                else:
                    grad_sum = (grad_values * grad_values).mean(1)
                    device = self.state_sum.device
                    if device != grad_indices.device:
                        grad_indices = grad_indices.to(device)
                    if device != grad_sum.device:
                        grad_sum = grad_sum.to(device)
                    self.lock.acquire()
                    self.state_sum.index_add_(0, grad_indices, grad_sum)
                    self.lock.release()
                    std = self.state_sum[grad_indices]  # _sparse_mask
                    if gpu_id >= 0:
                        std = std.cuda(gpu_id)
                    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                    tmp = (-clr * grad_values / std_values)
                    if tmp.device != device:
                        tmp = tmp.to(device)
                    # TODO(zhengda) the overhead is here.
                    self.lock.acquire()
                    self.emb.index_add_(0, grad_indices, tmp)
                    self.lock.release()
        self.trace = []

    def create_async_update(self):
        self.async_q = Queue(1)
        self.async_p = mp.Process(target=async_update, args=(None, self, self.async_q))
        self.async_p.start()

    def finish_async_update(self):
        self.async_q.put((None, None, None))
        self.async_p.join()

    def curr_emb(self):
        data = [data for _, data in self.trace]
        return th.cat(data, 0)

    def save(self, path, name):
        file_name = os.path.join(path, name+'.npy')
        np.save(file_name, self.emb.cpu().detach().numpy())

    def load(self, path, name):
        file_name = os.path.join(path, name+'.npy')
        self.emb = th.Tensor(np.load(file_name))
