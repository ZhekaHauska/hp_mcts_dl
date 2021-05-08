from mcts_dl.algorithms.mcts import MCTS
from mcts_dl.algorithms.value_policy_network import ValuePolicyNetwork
from mcts_dl.algorithms.model_network import ModelNetworkBorder

import torch
import os


class Agent:
    def __init__(self, config):
        self.config = config
        vp_config = config['vp_config']
        om_config = config['om_config']
        self.value_policy_net = ValuePolicyNetwork(**vp_config)
        self.observation_model_net = ModelNetworkBorder(**om_config)

        self.vp_checkpoint_name = config['vp_checkpoint']
        self.vp_checkpoint_path = config['vp_checkpoint_path']
        self.om_checkpoint_name = config['om_checkpoint']
        self.om_checkpoint_path = config['om_checkpoint_path']

        checkpoint = torch.load(os.path.join(self.vp_checkpoint_path, self.vp_checkpoint_name))
        self.value_policy_net.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(os.path.join(self.om_checkpoint_path, self.om_checkpoint_name))
        self.observation_model_net.load_state_dict(checkpoint['state_dict'])

        self.mcts = MCTS(config['mcts'])

    def make_action(self):
        pass
