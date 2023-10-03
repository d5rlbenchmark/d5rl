import numpy as np
from flax.training.train_state import TrainState

from jaxrl2.agents.common import eval_actions_jit, eval_log_prob_jit, sample_actions_jit
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import PRNGKey

from flax.training import checkpoints ###===### ###---###


class Agent(object):
    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
            self._actor.apply_fn, self._actor.params, observations
        )

        return np.asarray(actions)

    def eval_log_probs(self, batch: DatasetDict) -> float:
        return eval_log_prob_jit(self._actor.apply_fn, self._actor.params, batch)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng, self._actor.apply_fn, self._actor.params, observations
        )

        self._rng = rng

        return np.asarray(actions)

    ###===###
    @property
    def _save_dict(self):
        raise NotImplementedError

    def save_checkpoint(self, dir, step, keep_every_n_steps):
        checkpoints.save_checkpoint(dir, self._save_dict, step, prefix='checkpoint', overwrite=False, keep_every_n_steps=keep_every_n_steps)

    def restore_checkpoint(self, dir):
        raise NotImplementedError
    ###---###
