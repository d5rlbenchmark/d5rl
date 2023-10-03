import ml_collections
from ml_collections.config_dict import config_dict


def get_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 1e-3
    config.hidden_dims = (256, 256)
    config.cosine_decay = True
    config.dropout_rate = 0.1
    config.weight_decay = config_dict.placeholder(float)
    config.apply_tanh = True

    config.distr = "unitstd_normal"
    # unitstd_normal | tanh_normal | ar_mog

    return config

# def get_idql_config():
#     config = ml_collections.ConfigDict()

#     config.actor_lr = 3e-4
#     config.critic_lr=3e-4
#     config.value_lr=3e-4
#     config.T=5
#     config.N=64
#     config.M=0
#     config.actor_dropout_rate=0.1
#     config.actor_num_blocks=3
#     config.actor_weight_decay=None
#     config.actor_tau=0.001
#     config.actor_architecture='ln_resnet'
#     # config.critic_objective='expectile'
#     config.beta_schedule='vp'
#     # config.actor_objective='bc'
#     config.decay_steps=int(2e6) #Change this to int(4e6) for (2) (because you are finetuning actor)
#     config.actor_layer_norm=True

#     return config

def get_idql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.cosine_decay = True

    # # config.encoder = "resnet"
    # config.encoder = "impala"
    # config.use_multiplicative_cond = False

    config.tau = 0.005
    config.use_layer_norm = True
    config.dropout_rate = 0.1

    return config


def get_ddpm_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.cosine_decay = True

    # # config.encoder = "resnet"
    # config.encoder = "impala"
    # config.use_multiplicative_cond = False

    config.tau = 0.005
    config.use_layer_norm = True
    config.dropout_rate = 0.1

    return config

def get_config(config_string):
    possible_structures = {
        "bc": ml_collections.ConfigDict(
            {"model_constructor": "BCLearner", "model_config": get_bc_config()}
        ),

        "idql": ml_collections.ConfigDict(
            {"model_constructor": "DDPMIQLLearner", "model_config": get_idql_config()}
        ),

        "ddpm_bc": ml_collections.ConfigDict(
            {"model_constructor": "DDPMBCLearner", "model_config": get_idql_config()}
        ),
        
    }
    return possible_structures[config_string]
