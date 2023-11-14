from dataclasses import dataclass
from typing import Any
from typing import Optional

#from tianshou.data import ReplayBuffer


@dataclass
class BaseMsg:
    """A base message"""

    data: Optional[Any] = None


@dataclass
class InitMsg(BaseMsg):
    """Message a worker sends on startup"""

    pass


@dataclass
class BufferMsg(BaseMsg):
    """A replay buffer message sent from a worker"""

    def __post_init__(self):
        # assert isinstance(self.data, ReplayBuffer)
        assert True


@dataclass
class EvalResultsMsg(BaseMsg):
    """An evaluation results message sent from a worker"""

    def __post_init__(self):
        assert isinstance(self.data, dict)


@dataclass
class PolicyMsg(BaseMsg):
    """An RL policy message sent from a learner"""

    def __post_init__(self):
        assert isinstance(self.data, dict)
        assert "policy_id" in self.data
        assert "policy" in self.data

# For 'distTrain' training paradigm
@dataclass
class ParameterMsg(BaseMsg):
    """A parameter message sent from a worker"""

    def __post_init__(self):
        assert isinstance(self.data, dict)
        assert "parameters" in self.data