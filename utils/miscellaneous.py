import copy
import gc
import gzip
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
        self.payoffs = np.array([])
        self._n = 0

    @staticmethod
    def _make_arm(action_ids):
        arms = []
        for action_id in action_ids:
            arm = action.Action(action_id)
            arms.append(arm)
        return arms

    def learn_policy(self, data_parser):
        np.random.seed(self.seed)
        while True:
            try:
                current_arm_ids, context, logged_arm_id, payoff = next(data_parser)
                if self._action_storage.count():
                    previous_arm_ids = set(self._action_storage.iterids())
                    add_arm_ids = current_arm_ids.difference(previous_arm_ids)
                    remove_arm_ids = previous_arm_ids.difference(current_arm_ids)
                    if add_arm_ids:
                        current_arms = self._make_arm(add_arm_ids)
                        self.policy.add_action(current_arms)
                    if remove_arm_ids:
                        for remove_arm_id in remove_arm_ids:
                            self.policy.remove_action(remove_arm_id)
                else:
                    current_arms = self._make_arm(current_arm_ids)
                    self.policy.add_action(current_arms)
                history_id, recommendations = self.policy.get_action(context, 1)
                if logged_arm_id == recommendations[0].action.id:
                    if (self.data_size is None) or (np.random.uniform() <= self.data_size):
                        self.policy.reward(history_id, {recommendations[0].action.id: payoff})
                        self.payoffs = np.append(self.payoffs, np.array([payoff]))
                self._n += 1
                if (self.reset_freq is not None) and (self._n % self.reset_freq == 0):
                    del self.policy._history_storage
                    _ = gc.collect()
                    self.policy._history_storage = history.MemoryHistoryStorage()
                if (self.max_iter is not None) and (self._n >= self.max_iter):
                    break
            except StopIteration:
                break


def calculate_avg_payoff(payoffs):
    return np.mean(payoffs)


def calculate_avg_payoff_seq(payoffs):
    return np.cumsum(payoffs) / np.cumsum(np.ones(payoffs.shape))


def search_param(logger, base_policy_learner, param_grid, base_data_parser, **kwargs):
    results = []
    keys, values = zip(*param_grid.items())
    for value_tuple in itertools.product(*values):
        start = datetime.now()
        param = dict(zip(keys, value_tuple))
        policy_learner = copy.deepcopy(base_policy_learner)
        for key, value in param.items():
            if hasattr(policy_learner, key):
                setattr(policy_learner, key, value)
            else:
                setattr(policy_learner.policy, key, value)
        data_parser = base_data_parser(**kwargs)
        policy_learner.learn_policy(data_parser)
        results.append([param, calculate_avg_payoff(policy_learner.payoffs),
                        calculate_avg_payoff_seq(policy_learner.payoffs)])
        delta = datetime.now() - start
        logger.info('Parameters: {0}\tAverage Payoff: {1: 0.4%}\tRunning Time: {2}'.format(param, calculate_avg_payoff(
            policy_learner.payoffs), str(delta)))
        del policy_learner, data_parser
        _ = gc.collect()
    params, avg_payoffs, avg_payoff_seqs = zip(*results)
    return {'param': params, 'avg_payoff': avg_payoffs, 'avg_payoff_seq': avg_payoff_seqs}
