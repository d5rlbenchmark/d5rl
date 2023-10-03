import ml_collections
from ml_collections.config_dict import config_dict


def get_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.cnn_groups = 3 ###===### ###---###
    config.latent_dim = 50

    # config.encoder = "d4pg"
    # config.encoder = "resnet_34_v1"
    # config.encoder = "resnet_18_v1"
    config.encoder = "impala"

    config.encoder_norm = 'group'
    config.use_spatial_softmax = False
    config.softmax_temperature = -1,
    config.use_multiplicative_cond = False
    config.use_spatial_learned_embeddings = True

    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    return config

def get_ddpm_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4

    config.encoder = "resnet"

    config.cosine_decay = True
    config.use_layer_norm = True 
    config.dropout_rate = 0.1

    config.dropout_rate = config_dict.placeholder(float)

    return config

def get_idql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)

    #config.cnn_features = (16, 32, 64, 128, 256)
    #config.cnn_filters = (3, 3, 3, 3, 3)
    #config.cnn_strides = (2, 2, 2, 2, 2)
    #config.cnn_padding = "VALID"
    #config.cnn_groups = 3 ###===### ###---###
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.cosine_decay = True

    config.encoder = "resnet"

    config.tau = 0.005
    config.use_layer_norm = True 
    config.dropout_rate = 0.1

    return config

def get_iql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.cnn_groups = 3 ###===### ###---###
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.A_scaling = 3.0
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = "min"
    config.share_encoder = False

    return config

def get_cql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "valid"
    config.cnn_groups = 3 ###===### ###---###
    config.latent_dim = 50

    config.discount = 0.99

    config.cql_alpha = 5.0
    config.backup_entropy = False
    config.target_entropy = None
    config.init_temperature = 1.0
    config.max_q_backup = False
    config.dr3_coefficient = 0.0
    config.use_sarsa_backups = False
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = "min"
    config.share_encoder = False

    return config

def get_calql_config():

    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "valid"
    config.cnn_groups = 3 ###===### ###---###
    config.latent_dim = 50

    config.discount = 0.99

    config.cql_alpha = 5.0
    config.backup_entropy = False
    config.target_entropy = None
    config.init_temperature = 1.0
    config.max_q_backup = False
    config.dr3_coefficient = 0.0
    # config.use_sarsa_backups = False #
    config.dropout_rate = config_dict.placeholder(float)
    # config.cosine_decay = True #

    config.policy_encoder_type = "d4pg"
    config.encoder_type = "d4pg"
    # config.policy_encoder_type = "small"
    # config.encoder_type = "small"

    config.tau = 0.005

    config.critic_reduction = "min"
    config.share_encoders = False
    config.freeze_encoders = False

    return config

def get_adroit_bc_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.cnn_groups = 1 ###===### ###---###
    config.latent_dim = 50

    config.encoder = "d4pg"



    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    return config


def get_adroit_iql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.value_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.cnn_groups = 1 ###===### ###---###
    config.latent_dim = 50

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.A_scaling = 3.0
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = "min"
    config.share_encoder = False

    return config

def get_adroit_cql_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "valid"
    config.cnn_groups = 1 ###===### ###---###
    config.latent_dim = 50

    config.discount = 0.99

    config.cql_alpha = 5.0
    config.backup_entropy = False
    config.target_entropy = None
    config.init_temperature = 1.0
    config.max_q_backup = False
    config.dr3_coefficient = 0.0
    config.use_sarsa_backups = False
    config.dropout_rate = config_dict.placeholder(float)
    config.cosine_decay = True

    config.tau = 0.005

    config.critic_reduction = "min"
    config.share_encoder = False

    return config


def get_adroit_calql_config():

    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (16, 32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2, 2)
    config.cnn_padding = "valid"
    config.cnn_groups = 1 ###===### ###---###
    config.latent_dim = 50

    config.discount = 0.99

    config.cql_alpha = 5.0
    config.backup_entropy = False
    config.target_entropy = None
    config.init_temperature = 1.0
    config.max_q_backup = False
    config.dr3_coefficient = 0.0
    # config.use_sarsa_backups = False #
    config.dropout_rate = config_dict.placeholder(float)
    # config.cosine_decay = True #

    config.policy_encoder_type = "d4pg"
    config.encoder_type = "d4pg"
    # config.policy_encoder_type = "small"
    # config.encoder_type = "small"

    config.tau = 0.005

    config.critic_reduction = "min"
    config.share_encoders = False
    config.freeze_encoders = False

    return config

def get_config(config_string):
    possible_structures = {
        "bc": ml_collections.ConfigDict(
            {"model_constructor": "PixelBCLearner", "model_config": get_bc_config()}
        ),
        "ddpm_bc": ml_collections.ConfigDict(
            {"model_constructor": "PixelDDPMBCLearner", "model_config": get_ddpm_bc_config()}
        ),
        "idql": ml_collections.ConfigDict(
            {"model_constructor": "PixelIDQLLearner", "model_config": get_idql_config()}
        ),
        "iql": ml_collections.ConfigDict(
            {"model_constructor": "PixelIQLLearner", "model_config": get_iql_config()}
        ),
        "cql": ml_collections.ConfigDict(
            {"model_constructor": "PixelCQLLearner", "model_config": get_cql_config()}
        ),
        "calql": ml_collections.ConfigDict(
            {"model_constructor": "PixelCQLLearnerEncoderSepParallel", "model_config": get_calql_config()}
        ),
        "bc_adroit": ml_collections.ConfigDict(
            {"model_constructor": "PixelBCLearner", "model_config": get_adroit_bc_config()}
        ),
        "iql_adroit": ml_collections.ConfigDict(
            {"model_constructor": "PixelIQLLearner", "model_config": get_adroit_iql_config()}
        ),
        "cql_adroit": ml_collections.ConfigDict(
            {"model_constructor": "PixelCQLLearner", "model_config": get_adroit_cql_config()}
        ),
        "calql_adroit": ml_collections.ConfigDict(
            {"model_constructor": "PixelCQLLearnerEncoderSepParallel", "model_config": get_adroit_calql_config()}
        ),
    }
    return possible_structures[config_string]
