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
        self._num_preds, self._num_fits = 0, 0

    @staticmethod
    def _make_arm(arm_ids):
        arms = []
        for arm_id in arm_ids:
            arm = action.Action(arm_id)
            arms.append(arm)
        return arms

    def learn_policy(self, data_parser):
        self._num_preds = 0
        np.random.seed(self.seed)
        while True:
            try:
                timestamp, current_arm_ids, context, logged_arm_id, payoff = next(data_parser)
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
                    self._num_fits += 1
                    if (self.reset_freq is not None) and (self._num_fits % self.reset_freq == 0):
                        self._history_storage = history.MemoryHistoryStorage()
                self._num_preds += 1
                if (self.max_iter is not None) and (self._num_preds >= self.max_iter):
                    break
            except StopIteration:
                break


def calculate_avg_payoff(payoffs):
    return np.mean(payoffs)


def calculate_avg_payoff_seq(payoffs):
    return np.cumsum(payoffs) / np.cumsum(np.ones(payoffs.shape))


def parse_data_from_file(data_file_paths):
    for data_file_path in data_file_paths:
        with gzip.open(data_file_path, 'rt') as file:
            for line in file:
                chunks = line.split('|')
                feature = {}
                timestamp, logged_arm_id, payoff = None, None, None
                for i, chunk in enumerate(chunks):
                    values = chunk.rstrip().split(' ')
                    if i == 0:
                        timestamp, logged_arm_id, payoff = values[0], int(values[1]), int(values[2])
                    else:
                        feature[values[0]] = np.array(list(map(lambda x: float(x.split(':')[1]), sorted(values[1:]))))
                arm_ids = set(feature.keys())
                arm_ids = set(map(lambda x: int(x), arm_ids.difference({'user'})))
                context = {arm_id: feature['user'] for arm_id in arm_ids}
                yield timestamp, arm_ids, context, logged_arm_id, payoff


def search_param(logger, base_policy_learner, param_grid, base_data_parser, *args):
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
        data_parser = base_data_parser(*args)
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