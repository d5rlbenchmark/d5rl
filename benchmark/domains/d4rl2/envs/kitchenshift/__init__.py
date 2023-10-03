import os

from gym.envs.registration import register

register('kitchen-v1',
         # entry_point="d4rl2.wrappers.offline_env:make_env",
         entry_point="benchmark.domains.d4rl2.wrappers.offline_env:make_env",
         max_episode_steps=500,
         kwargs=dict(task='kitchen-v1',
                     tasks_to_complete = ['microwave', 'kettle', 'switch', 'slide'],
                     datasets = ['expert_demos', 'expert_suboptimal', 'play_data']))

register('random_kitchen-v1',
         # entry_point="d4rl2.wrappers.offline_env:make_env",
         entry_point="benchmark.domains.d4rl2.wrappers.offline_env:make_env",
         max_episode_steps=500,
         kwargs=dict(task='random_kitchen-v1',
                     tasks_to_complete = ['microwave', 'kettle', 'switch', 'slide'],
                     datasets = ['expert_demos', 'expert_suboptimal', 'play_data']))

# register('random_kitchen_indistribution_expert-v1',
#          # entry_point="d4rl2.wrappers.offline_env:make_env",
#          entry_point="benchmark.domains.d4rl2.wrappers.offline_env:make_env",
#          max_episode_steps=500,
#          # microwave+kettle+light switch+slide cabinet
#          kwargs=dict(task='random_kitchen-v1',
#                      tasks_to_complete = ['microwave', 'kettle', 'switch', 'slide'],
#                      datasets = ['expert_demos']))
#
# register('random_kitchen_indistribution_play-v1',
#          # entry_point="d4rl2.wrappers.offline_env:make_env",
#          entry_point="benchmark.domains.d4rl2.wrappers.offline_env:make_env",
#          max_episode_steps=500,
#          # microwave+kettle+light switch+slide cabinet
#          kwargs=dict(task='random_kitchen-v1',
#                      tasks_to_complete = ['microwave', 'kettle', 'switch', 'slide'],
#                      datasets = ['play_data']))
#
# register('random_kitchen_outofdistribution_expert-v1',
#          # entry_point="d4rl2.wrappers.offline_env:make_env",
#          entry_point="benchmark.domains.d4rl2.wrappers.offline_env:make_env",
#          max_episode_steps=500,
#          # microwave+kettle+bottom burner+light switch
#          kwargs=dict(task='random_kitchen-v1',
#                      tasks_to_complete = ['microwave', 'kettle', "bottomknob", 'switch'],
#                      datasets = ['expert_demos']))
#
# register('random_kitchen_outofdistribution_play-v1',
#          # entry_point="d4rl2.wrappers.offline_env:make_env",
#          entry_point="benchmark.domains.d4rl2.wrappers.offline_env:make_env",
#          max_episode_steps=500,
#          # microwave+kettle+bottom burner+light switch
#          kwargs=dict(task='random_kitchen-v1',
#                      tasks_to_complete = ['microwave', 'kettle', "bottomknob", 'switch'],
#                      datasets = ['play_data']))
