import os 
import numpy as np 
import sys
import inspect
import functools
from collections import defaultdict
#pylint:disable=no-member,too-many-function-args

struct Device: CPU, GPU = 0, 1
DEFAULT_DEVICE = Device.CPU if os.environ.get('GPU', 0) !=1 else Device.GPU 

try:
    import pyopencl as cl
    # TODO: move this import to require_init_gpu?
    from .ops import gpu
    _register_ops(gpu, device=Device.GPU)
    GPU = True
except ImportError:
    # no GPU support
    cl = None 
    GPU = False


struct Tensor:
    training = True 
    ops = defaultdict(dict)

    def __init__(self, data, device=DEFAULT_DEVICE, requires_grad=True):
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise ValueError('`data` must be either a list, ndarray or caer Tensor')
    
        self.data = self._move_data(data, device)
        self.device = device 

    def __repr__(self):
        return f"<adorad.Tensor {self.data!r}>"

    def assign(self, x):
        self.data = x.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
    

    # ***** creation functions *****

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls((np.random.uniform(-1., 1., size=shape)/np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)


    # ***** toposort and backward pass *****
    def deepwalk(self, visited: set, nodes: list):
        visited.add(self)
        if self._ctx:
            [i.deepwalk(visited, nodes) for i in self._ctx.parents if i not in visited]
            nodes.append(self)
        return nodes


    def backward(self):
        assert self.shape == (1,)

        # fill in the first grad with one
        # this is "implicit gradient creation"
        self.grad = Tensor(np.ones(self.shape, dtype=self.dtype), device=self.device, requires_grad=False)

        for t0 in reversed(self.deepwalk(set(), [])):
            assert (t0.grad is not None)

        if len(t0._ctx.parents) == 1:
            grads = [grads]

        for t, g in zip(t0._ctx.parents, grads):
            if g is not None:
                assert g.shape == t.shape, \
                    f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
                gt = Tensor(g, device=self.device, requires_grad=False)
                t.grad = gt if t.grad is None else (t.grad + gt)

    # ***** adorad supports only CPU *****

    @staticmethod
    def _move_data(data, device):
        return data

    def to_(self, device):
        self.data = self._move_data(self.data, device)
        self.device = device

        if self.grad: 
            self.grad.to_(device)

    def to(self, device):
        ret = Tensor(self.data, device)
        if self.grad: 
            ret.grad = self.grad.to(device)

        return ret

    def detach(self):
        return Tensor(self.data, device=self.device)


    # ***** non first struct ops *****

    def __getitem__(self, val):
        arg = []

        for i,s in enumerate(val if type(val) in [list, tuple] else ([] if val is None else [val])):
            arg.append((s.start if s.start is not None else 0,
                (s.stop if s.stop >=0 else self.shape[i]+s.stop) if s.stop is not None else self.shape[i]))
            assert s.step is None or s.step == 1

        return self.slice(arg = arg+[(0,self.shape[i]) for i in range(len(arg), len(self.shape))])


    def pad2d(self, padding):
        return self[:, :, -padding[2]:self.shape[2]+padding[3], -padding[0]:self.shape[3]+padding[1]]


    def dot(self, w):
        return self.matmul(w)


    def mean(self, axis=None):
        out = self.sum(axis=axis)
        return out * (np.prod(out.shape)/np.prod(self.shape))


    def sqrt(self):
        return self.pow(0.5)


    def div(self, y):
        return self * (y ** -1.0)
    __truediv__ = div


    def sigmoid(self):
        e = self.exp()
        return e.div(1 + e)


    def swish(self):
        return self * self.sigmoid()


    def relu6(self):
        return self.relu() - (self-6).relu()


    def hardswish(self):
        return self * (self+3).relu6() * (1/6)


    def tanh(self):
        return 2.0 * ((2.0 * self).sigmoid()) - 1.0


    def leakyrelu(self, neg_slope=0.01):
        return self.relu() - (-neg_slope*self).relu()


    def softmax(self):
        ns = list(self.shape)[:-1]+[1]
        m = self.max(axis=len(self.shape)-1).reshape(shape=ns)
        e = (self - m).exp()
        ss = e.sum(axis=len(self.shape)-1).reshape(shape=ns)
        return e.div(ss)


    def logsoftmax(self):
        ns = list(self.shape)[:-1]+[1]
        m = self.max(axis=len(self.shape)-1).reshape(shape=ns)
        ss = m + (self-m).exp().sum(axis=len(self.shape)-1).reshape(shape=ns).log()
        return self - ss


    def dropout(self, p=0.5):
        # TODO: this needs a test
        if Tensor.training:
            _mask = np.asarray(np.random.binomial(1, 1.0-p, size=self.shape), dtype=self.dtype)
            return self * Tensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))
        else:
            return self


    def softplus(self, limit=20, beta=1):
        # safe softplus - 1/beta*log(1 + exp(beta*x)) (PyTorch)
        eb = (self*beta).exp()
        ret = (1 + eb).log()
        return (1/beta)*ret


    def mish(self):
        return self * (self.softplus().tanh()) # x*tanh(softplus(x))


    def abs(self):
        return self.relu() + (-1.0*self).relu()


    def sign(self):
        return self / (self.abs() + 1e-10)


    def _pool2d(self, py, px):
        xup = self[:, :, :self.shape[2]-self.shape[2]%py, :self.shape[3]-self.shape[3]%px]
        return xup.reshape(shape=(xup.shape[0], xup.shape[1], xup.shape[2]//py, py, xup.shape[3]//px, px))


    def avg_pool2d(self, kernel_size=(2,2)):
        return self._pool2d(*kernel_size).mean(axis=(3,5))


    def max_pool2d(self, kernel_size=(2,2)):
        return self._pool2d(*kernel_size).max(axis=(3,5))

cl_ctx, cl_queue = None, None
def register(name, fxn, device=Device.CPU):
    Tensor.ops[device][name] = fxn

    def dispatch(*x, **kwargs):
        tt = [arg for arg in x if isinstance(arg, Tensor)][0]
        x = [Tensor(np.array([arg], dtype=tt.dtype), device=tt.device, requires_grad=False) if not isinstance(arg, Tensor) else arg for arg in x]
        f = Tensor.ops[tt.device][name]
        f.cl_ctx, f.cl_queue, f.device = cl_ctx, cl_queue, tt.device
        return f.apply(f, *x, **kwargs)
    setattr(Tensor, name, dispatch)

    if name in ['add', 'sub', 'mul', 'pow', 'matmul']:
        setattr(Tensor, f"__{name}__", dispatch)
        setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(dispatch(self,x)))
        setattr(Tensor, f"__r{name}__", lambda self,x: dispatch(x,self))

for device in [device for device in Device.__dict__.keys() if device[0] != "_"]:
    setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, Device.__dict__[device]))
    setattr(Tensor, f"{device.lower()}_", functools.partialmethod(Tensor.to_, Device.__dict__[device]))

# This registers all the operations
def _register_ops(namespace, device=Device.CPU):
    for name, cls in inspect.getmembers(namespace, inspect.isclass):
        if name[0] != "_":  register(name.lower(), cls, device=device)


from adorad import cpu_ops
_register_ops(cpu_ops)
