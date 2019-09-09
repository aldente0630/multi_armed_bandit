import copy
import gc
import itertools
import numpy as np
from datetime import datetime
from striatum.storage import history
from striatum.storage import model
from striatum.storage import action


class PolicyLearner:
    def __init__(self, policy, **params):
        self.seed = None
        self.data_size = None
        self.max_iter = None
        self.reset_freq = None
        if 'seed' in params:
            self.seed = params.pop('seed')
        if 'data_size' in params:
            self.data_size = params.pop('data_size')
        if 'max_iter' in params:
            self.max_iter = params.pop('max_iter')
        if 'reset_freq' in params:
            self.reset_freq = params.pop('reset_freq')
        self._params = params
        self._history_storage = history.MemoryHistoryStorage()
        self._model_storage = model.MemoryModelStorage()
        self._action_storage = action.MemoryActionStorage()
        self.policy = policy(self._history_storage, self._model_storage, self._action_storage, **self._params)
        self.reward_values = np.array([])
        self._n_iter = 0

    @staticmethod
    def _make_action(action_ids):
        actions = []
        for action_id in action_ids:
            actions.append(action.Action(action_id))
        return actions

    def learn_policy(self, data_parser):
        np.random.seed(self.seed)
        while True:
            try:
                _, context, rewards = next(data_parser)
                curr_action_ids = set(context.keys())
                if self._action_storage.count():
                    prev_action_ids = set(self._action_storage.iterids())
                    new_action_ids = curr_action_ids.difference(prev_action_ids)
                    old_action_ids = prev_action_ids.difference(curr_action_ids)
                    if new_action_ids:
                        curr_actions = self._make_action(new_action_ids)
                        self.policy.add_action(curr_actions)
                    if old_action_ids:
                        for old_action_id in old_action_ids:
                            self.policy.remove_action(old_action_id)
                else:
                    curr_actions = self._make_action(curr_action_ids)
                    self.policy.add_action(curr_actions)
                obs_action_id = max(rewards, key=rewards.get)
                history_id, recommendations = self.policy.get_action(context, 1)
                if obs_action_id == recommendations[0].action.id:
                    if (self.data_size is None) or (np.random.uniform() <= self.data_size):
                        self.policy.reward(history_id, rewards)
                        self.reward_values = np.append(self.reward_values, np.array(rewards[obs_action_id]))
                self._n_iter += 1
                if (self.reset_freq is not None) and (self._n_iter % self.reset_freq == 0):
                    del self.policy._history_storage
                    _ = gc.collect()
                    self.policy._history_storage = history.MemoryHistoryStorage()
                if (self.max_iter is not None) and (self._n_iter >= self.max_iter):
                    break
            except StopIteration:
                break


def get_avg_reward(reward_values):
    return np.mean(reward_values)


def get_seq_avg_reward(reward_values):
    return np.cumsum(reward_values) / np.cumsum(np.ones(reward_values.shape))


def search_param(base_policy_learner, param_grid, base_data_parser, **kwargs):
    res = []
    keys, values = zip(*param_grid.items())
    for prod in itertools.product(*values):
        start = datetime.now()
        param = dict(zip(keys, prod))
        policy_learner = copy.deepcopy(base_policy_learner)
        for key, value in param.items():
            if hasattr(policy_learner, key):
                setattr(policy_learner, key, value)
            else:
                setattr(policy_learner.policy, key, value)
        data_parser = base_data_parser(**kwargs)
        policy_learner.learn_policy(data_parser)
        res.append([param, get_avg_reward(policy_learner.reward_values), 
                    get_seq_avg_reward(policy_learner.reward_values)])
        print('Parameters: {0} | Average Reward: {1:0.4%} | Computation Time: {2}'.format(
            param, get_avg_reward(policy_learner.reward_values), str(datetime.now() - start).split('.')[0]))
        del policy_learner, data_parser
        _ = gc.collect()
    params, avg_rewards, seq_avg_rewards = zip(*res)
    return {'param': params, 'avg_reward': avg_rewards, 'seq_avg_reward': seq_avg_rewards}
