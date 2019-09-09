import gzip
import numpy as np


def parse_data(data_paths, share_context=False, is_shared_context_first=True):
    for data_path in data_paths:
        with gzip.open(data_path, 'rt') as file:
            for line in file:
                chunks = line.split('|')
                rewards = {}
                features = {}
                for idx, chunk in enumerate(chunks):
                    values = chunk.rstrip().split(' ')
                    if idx == 0:
                        timestamp = str(values[0])
                        rewards[int(values[1])] = float(values[2])
                    elif len(values) == 7:
                        key = -1 if values[0] == 'user' else int(values[0])
                        features[key] = np.array(list(map(lambda x: float(x.split(':')[1]), sorted(values[1:]))))
                action_ids = list(set(features.keys()).difference({-1}))
                if not share_context:
                    context = {action_id: features[-1] for action_id in action_ids}
                elif is_shared_context_first:
                    context = {action_id: np.append(np.outer(features[-1], features[action_id]).flatten()[1:], 
                                                    features[-1]) for action_id in action_ids}
                else:
                    context = {action_id: np.append(features[-1], np.outer(
                        features[-1], features[action_id]).flatten()[1:]) for action_id in action_ids}
                yield timestamp, context, rewards
