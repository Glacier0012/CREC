from torch.optim import Adam
from crec.config import LazyCall

optim = LazyCall(Adam)(
    # optim.params is meant to be set before instantiating
    lr=0.0001,
    betas=(0.9, 0.98),
    eps=1e-9
)