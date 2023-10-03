from absl import app, flags

import d4rl2.envs.a1
from d4rl2.envs.a1.collect.maze.collect_utils import collect_parallel

FLAGS = flags.FLAGS

flags.DEFINE_enum('maze_name', 'umaze', ['umaze', 'medium_maze'], 'Maze name.')
flags.DEFINE_integer('num_samples', 1000000, 'Number of samples.')
flags.DEFINE_integer('seed', 42, 'Seed.')


def main(_):
    collect_parallel(FLAGS.maze_name,
                     FLAGS.num_samples,
                     FLAGS.seed,
                     exclude_expert=True)


if __name__ == '__main__':
    app.run(main)
