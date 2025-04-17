from torch import nn
from configs.hyperparams import HYPERPARAMS

def get_policy_kwargs():
    """Return policy network architecture configuration"""
    return {
        "features_extractor_class": None,
        "activation_fn": nn.ReLU,
        "net_arch": dict(
            pi=HYPERPARAMS["ppo"]["policy_kwargs"]["net_arch"]["pi"],
            vf=HYPERPARAMS["ppo"]["policy_kwargs"]["net_arch"]["vf"]
        )
    }