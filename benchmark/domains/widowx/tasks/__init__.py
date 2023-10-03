import inspect

from dm_control.composer.environment import Environment
from dm_control.composer.task import Task
from dm_control.suite.wrappers import action_scale, pixels
from dm_env_wrappers import ConcatObservationWrapper, SinglePrecisionWrapper

from benchmark.domains.widowx.tasks.base_task import BaseTask
from benchmark.domains.widowx.tasks.drawer import Drawer
from benchmark.domains.widowx.tasks.pick_and_place import PickAndPlace
from benchmark.domains.widowx.tasks.reach import Reach, ReachSparse

TASKS = {
    name: dclass
    for name, dclass in locals().items()
    if (
        (inspect.isclass(dclass) and issubclass(dclass, Task))
        or (hasattr(dclass, "func") and issubclass(dclass.func, Task))
    )
}


def load(task_name: str):
    task = TASKS[task_name]()

    env = Environment(
        task=task,
        strip_singleton_obs_buffer_dim=True,
        raise_exception_on_physics_error=False,
    )
    env = action_scale.Wrapper(env, -1, 1)
    env = SinglePrecisionWrapper(env)
    env = ConcatObservationWrapper(env)
    env = pixels.Wrapper(
        env,
        pixels_only=False,
        render_kwargs={"camera_id": "wx250s/pixels", "height": 84, "width": 84},
    )

    return env
