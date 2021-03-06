import torch
import numpy
import math

from mcts_dl.environment.gridworld_pomdp import is_move_possible


class Node:
    def __init__(self, prior):
        """

        :param prior:
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.observation = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, policy_logits, observation):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.observation = observation

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)


class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.pb_c_base = config['pb_c_base']
        self.pb_c_init = config['pb_c_init']
        self.config = config
        self.num_simulations = config['num_simulations']
        self.discount = config['discount']
        self.actions = numpy.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
                                    [0, 1], [0, -1], [1, 0], [-1, 0]])

    def run(
        self,
        policy_value_model,
        observation_model,
        observation,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            (root_predicted_value, policy_logits) = policy_value_model(*observation)
            legal_actions = self.get_legal_actions(observation[0].squeeze().cpu().numpy())
            root.expand(legal_actions, policy_logits, observation)

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)
            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            # get next observation
            win_obs = parent.observation[0].squeeze().cpu().numpy()
            vec_obs = parent.observation[1].squeeze().cpu().numpy()
            observation = observation_model.run(
                (win_obs, vec_obs),
                torch.from_numpy(self.actions[action]),
                parent.observation[0].device,
            )
            # get value and policy for this new observation
            next_win_obs = observation[0]
            next_win_obs = torch.from_numpy(next_win_obs).float().unsqueeze(0).unsqueeze(0).to(parent.observation[0].device)
            next_vec_obs = torch.from_numpy(observation[1]).float().unsqueeze(0).to(parent.observation[0].device)
            value, policy_logits = policy_value_model(next_win_obs, next_vec_obs)
            value = value.item()
            # define legal actions
            legal_actions = self.get_legal_actions(observation[0])
            node.expand(
                legal_actions,
                policy_logits,
                (next_win_obs, next_vec_obs),
            )

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.pb_c_base + 1) / self.pb_c_base
            )
            + self.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(self.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(self.discount * node.value())

            value = self.discount * value

    def get_legal_actions(self, observation_window):
        legal_actions = list()
        for action, displacement in enumerate(self.actions):
            if is_move_possible(observation_window, displacement):
                legal_actions.append(action)
        return legal_actions


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
