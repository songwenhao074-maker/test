"""Protocol016: shared BWGD2 demand adapter with training-derived synthetic disk."""
import copy
import json
from pathlib import Path
import numpy as np
from .BitbrainWorkload2 import BWGD2

ROOT = Path(__file__).resolve().parents[2]
LAW_PATH = ROOT/'artifacts/ftmoe_online/adapted_bwgd2_016/disk_law.json'


def map_cpu(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid CPU demand')
    return np.where(x > 0, np.clip(x, 2., 1860.), 0.)


class TrainingMarkovDisk:
    """Synthetic storage occupancy, indexed by task age; getters are idempotent."""
    def __init__(self, law, replay_seed, creation_id):
        self.values = np.asarray(law['values'], dtype=float)
        self.transition = np.asarray(law['transition'], dtype=float)
        self.generator = np.random.default_rng(np.random.SeedSequence([replay_seed, creation_id, 16016]))
        self.states = [int(self.generator.choice(len(self.values), p=law['initial_probability']))]

    def allocContainer(self, container):
        self.container = container

    def disk(self):
        age = self.container.env.interval - self.container.startAt
        if age < 0:
            raise ValueError('Negative container age')
        while len(self.states) <= age:
            self.states.append(int(self.generator.choice(len(self.values), p=self.transition[self.states[-1]])))
        return float(self.values[self.states[age]]), 1., 1.

    def save(self):
        return {'states': list(self.states), 'rng': copy.deepcopy(self.generator.bit_generator.state)}

    def restore(self, state):
        self.states = list(state['states'])
        self.generator.bit_generator.state = copy.deepcopy(state['rng'])


class AdaptedBWGD2(BWGD2):
    def __init__(self, mean, sigma, replay_seed):
        super().__init__(mean, sigma)
        self.replay_seed = replay_seed
        self.disk_law = json.loads(LAW_PATH.read_text(encoding='utf8'))

    def adapt_new_tasks(self, first):
        # Only newly created tasks are transformed; queued deployment retries reuse the same models.
        for i in range(first, len(self.createdContainers)):
            cid, interval, ips, ram, _ = self.createdContainers[i]
            if ips.completedInstructions or ips.totalInstructions:
                raise AssertionError('Task already executed before adaptation')
            ips.ips_list = map_cpu(ips.ips_list).tolist()
            ips.max_ips = max(float(map_cpu([ips.max_ips])[0]), max(ips.ips_list))
            ram.size_list = (np.asarray(ram.size_list) * 2.).tolist()
            ram.read_list = [1.] * len(ram.read_list)
            ram.write_list = [1.] * len(ram.write_list)
            disk = TrainingMarkovDisk(self.disk_law, self.replay_seed, cid)
            self.createdContainers[i] = (cid, interval, ips, ram, disk)

    def generateNewContainers(self, interval):
        first = len(self.createdContainers)
        super().generateNewContainers(interval)
        self.adapt_new_tasks(first)
        return self.getUndeployedContainers()
