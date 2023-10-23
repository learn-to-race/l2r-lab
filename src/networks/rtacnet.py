from torch.nn import Linear, Sequential, ReLU, ModuleList, Module, Parameter
from torch.distributions import Distribution, Normal, Independent
from torch.nn.functional import leaky_relu
import torch 
from typing import TypeVar, Union, Type, Callable, Any, Dict, Sequence, Mapping
import numpy as np

def collate(batch, device=None):
  elem = batch[0]
  if isinstance(elem, torch.Tensor):
    return torch.stack(batch).to(device)
    # if elem.numel() < 20000:  # TODO: link to the relavant profiling that lead to this threshold
    #   return torch.stack(batch).to(device)
    # else:
    #   return torch.stack([b.contiguous().to(device) for b in batch], 0)
  elif isinstance(elem, np.ndarray):
    return collate(tuple(torch.from_numpy(b) for b in batch), device)
  elif hasattr(elem, '__torch_tensor__'):
    return torch.stack([b.__torch_tensor__().to(device) for b in batch], 0)
  elif isinstance(elem, Sequence):
    transposed = zip(*batch)
    return type(elem)(collate(samples, device) for samples in transposed)
  elif isinstance(elem, Mapping):
    return type(elem)((key, collate(tuple(d[key] for d in batch), device)) for key in elem)
  else:
    return torch.from_numpy(np.array(batch)).to(device)  # we create a numpy array first to work around https://github.com/pytorch/pytorch/issues/24200


def partition(x):
  if isinstance(x, torch.Tensor):
    # return x.cpu()
    return x.cpu().numpy()  # perhaps we should convert this to tuple for consistency?
  elif isinstance(x, Mapping):
    m = {k: partition(x[k]) for k in x}
    numel = len(tuple(m.values())[0])
    out = tuple(type(x)((key, value[i]) for key, value in m.items()) for i in range(numel))
    return out
  raise TypeError()


class TanhNormal(Distribution):
  """Distribution of X ~ tanh(Z) where Z ~ N(mean, std)
  Adapted from https://github.com/vitchyr/rlkit
  """
  def __init__(self, normal_mean, normal_std, epsilon=1e-6):
    self.normal_mean = normal_mean
    self.normal_std = normal_std
    self.normal = Normal(normal_mean, normal_std)
    self.epsilon = epsilon
    super().__init__(self.normal.batch_shape, self.normal.event_shape)

  def log_prob(self, x):
    assert hasattr(x, "pre_tanh_value")
    assert x.dim() == 2 and x.pre_tanh_value.dim() == 2
    return self.normal.log_prob(x.pre_tanh_value) - torch.log(
      1 - x * x + self.epsilon
    )

  def sample(self, sample_shape=torch.Size()):
    z = self.normal.sample(sample_shape)
    out = torch.tanh(z)
    out.pre_tanh_value = z
    return out

  def rsample(self, sample_shape=torch.Size()):
    z = self.normal.rsample(sample_shape)
    out = torch.tanh(z)
    out.pre_tanh_value = z
    return out

class TanhNormalLayer(torch.nn.Module):
  def __init__(self, n, m):
    super().__init__()

    self.lin_mean = torch.nn.Linear(n, m)
    # self.lin_mean.weight.data
    # self.lin_mean.bias.data

    self.lin_std = torch.nn.Linear(n, m)
    self.lin_std.weight.data.uniform_(-1e-3, 1e-3)
    self.lin_std.bias.data.uniform_(-1e-3, 1e-3)

  def forward(self, x):
    mean = self.lin_mean(x)
    log_std = self.lin_std(x)
    log_std = torch.clamp(log_std, -20, 2)
    std = torch.exp(log_std)
    # a = TanhTransformedDist(Independent(Normal(m, std), 1))
    a = Independent(TanhNormal(mean, std), 1)
    return a



def big_conv(n):
  # if input shape = 64 x 256 then output shape = 2 x 26
  return torch.nn.Sequential(
    torch.nn.Conv2d(n, 64, 8, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(64, 64, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(64, 128, 4, stride=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(128, 128, 4, stride=1), torch.nn.LeakyReLU(),
  )

class ActorModule(Module):
  actor: callable

  # noinspection PyMethodOverriding
  def to(self, device):
    self.device = device
    return super().to(device=device)

  def act(self, obs, r, done, info, train=False):
    obs_col = collate((obs,), device=self.device)
    with torch.no_grad():
      action_distribution = self.actor(obs_col)
      action_col = action_distribution.sample() if train else action_distribution.sample_deterministic()
    action, = partition(action_col)
    return action, []

class ConvRTAC(ActorModule):
  def __init__(self, observation_space, action_space, hidden_units: int = 512, Conv: type = big_conv):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    (img_sp, vec_sp), ac_sp = observation_space

    self.conv = Conv(img_sp.shape[0])

    with torch.no_grad():
      conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

    self.lin1 = Linear(conv_size + vec_sp.shape[0] + ac_sp.shape[0], hidden_units)
    self.lin2 = Linear(hidden_units + vec_sp.shape[0] + ac_sp.shape[0], hidden_units)
    self.critic_layer = Linear(hidden_units, 1)
    self.actor_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
    self.critic_output_layers = (self.critic_layer,)

  def forward(self, inp):
    (x, vec), action = inp
    x = x.type(torch.float32)
    x = x / 255 - 0.5
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = leaky_relu(self.lin1(torch.cat((x, vec, action), -1)))
    x = leaky_relu(self.lin2(torch.cat((x, vec, action), -1)))
    v = self.critic_layer(x)
    action_distribution = self.actor_layer(x)
    return action_distribution, (v,), (x,)

class ConvSeparate(ActorModule):
  def __init__(self, observation_space, action_space, hidden_units: int = 512, conv: type = big_conv):
    super().__init__()
    self.a = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)
    self.b = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)
    self.c = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)

  @property
  def critic_output_layers(self):
    return self.b.critic_output_layers + self.c.critic_output_layers

  def actor(self, x):
    return self.a(x)[0]

  def forward(self, x):
    action_distribution, *_ = self.a(x)
    _, v0, h0 = self.b(x)
    _, v1, h1 = self.c(x)
    return action_distribution, v0+v1, h0+h1



class PopArt(Module):
  """PopArt http://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude"""
  def __init__(self, output_layer, beta: float = 0.0003, zero_debias: bool = True, start_pop: int = 8):
    # zero_debias=True and start_pop=8 seem to improve things a little but (False, 0) works as well
    super().__init__()
    self.start_pop = start_pop
    self.beta = beta
    self.zero_debias = zero_debias
    self.output_layers = output_layer if isinstance(output_layer, (tuple, list, torch.nn.ModuleList)) else (output_layer,)
    shape = self.output_layers[0].bias.shape
    device = self.output_layers[0].bias.device
    assert all(shape == x.bias.shape for x in self.output_layers)
    self.mean = Parameter(torch.zeros(shape, device=device), requires_grad=False)
    self.mean_square = Parameter(torch.ones(shape, device=device), requires_grad=False)
    self.std = Parameter(torch.ones(shape, device=device), requires_grad=False)
    self.updates = 0

  @torch.no_grad()
  def update(self, targets):
    beta = max(1/(self.updates+1), self.beta) if self.zero_debias else self.beta
    # note that for beta = 1/self.updates the resulting mean, std would be the true mean and std over all past data

    new_mean = (1 - beta) * self.mean + beta * targets.mean(0)
    new_mean_square = (1 - beta) * self.mean_square + beta * (targets * targets).mean(0)
    new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(0.0001, 1e6)

    assert self.std.shape == (1,), 'this has only been tested in 1D'

    if self.updates >= self.start_pop:
      for layer in self.output_layers:
        # TODO: Properly apply PopArt in RTAC and remove the hack below
        # We modify the weight while it's gradient is being computed
        # Therefore we have to use .data (Pytorch would otherwise throw an error)
        layer.weight *= self.std / new_std
        layer.bias *= self.std
        layer.bias += self.mean - new_mean
        layer.bias /= new_std

    self.mean.copy_(new_mean)
    self.mean_square.copy_(new_mean_square)
    self.std.copy_(new_std)
    self.updates += 1
    return self.normalize(targets)

  def normalize(self, x):
    return (x - self.mean) / self.std

  def unnormalize(self, value):
    return value * self.std + self.mean