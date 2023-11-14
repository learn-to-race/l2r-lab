# Bug fixes
## l2r-distributed.yaml
1. Prepend `apt-get update` for worker-pods' command
2. Replace `worker.py` and `learner.py` with `distributedworker.py` and `distributedserver.py` respectively for worker-pods and learner-pod
3. Add `tensorboardX` to the installation command for learner-pod

## [distributed-l2r] branch - Still BUGS!
1. Moved `distributedworker.py` and `distributedlearner.py` from `./scripts/` to `root`
2. `./src/config/schema.py` missing fields from `config_files/async_sac/agent.yaml`
3. `./config_files/async_sac/agent.yaml` update parameter name `actor_critic_cfg -> actor_critic_cfg_path`
4. Added `state_dict(self)` function in `SACAgent.py`
5. Remove `entity="learn2race"` from `src/loggers/WanDBLogger.py`
6. Added `wandb tensorboardX jsonpickle` for the pip install of the server and learner in `l2r-distributed.yaml`
7. Using the `load_model()` from the `l2r-benchmarks -> distributed-aicrowd` in `SACAgent.yaml`

## [distributed-baselines] branch (`mountain-car` + `bipedal-walker`)
1. Remove `entity="learn2race"` from `src/loggers/WanDBLogger.py`

2. Change the default action space from `self.action_space = Box(-1, 1, (4,))` to `self.action_space = Box(-1, 1, (self.actor_critic.action_dim,))` in `src/agents/SACAgent.py`

3. Add `self.action_dim = action_dim` to `class ActorCritic(nn.Module)` in `src/networks/critic.py`, so that the above command can work

4. Add the handling of action being a scalar in `select_action()` in `src/agents/SACAgents.py`
```python
def select_action(self, obs):
    ...
    a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
    if a.shape == ():
        # In case a in a scalar
        a = np.array([a])
    action_obj.action = a
    ...
```

5. Modifications in `src/networks/network_baselines.py`
```py
# SquashedGaussianMLPActor()
def forward(self, obs, deterministic=False, with_logprob=True):
    net_out = self.net(obs)
    mu = self.mu_layer(net_out)
    log_std = self.log_std_layer(net_out)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)
    pi_distribution = Normal(mu, std)
    if deterministic:
        # Only used for evaluating policy at test time.
        pi_action = mu
    else:
        pi_action = pi_distribution.rsample()

    if with_logprob:
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
            axis=1
        )
    else:
        logp_pi = None

    pi_action = torch.tanh(pi_action)
    pi_action = self.act_limit * pi_action

    return pi_action, logp_pi
```

6. Add environment variable `AGENT_NAME` in both code and deployment YAML to configure which agent to run