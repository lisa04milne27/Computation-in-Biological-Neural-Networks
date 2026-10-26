from enum import Enum

import jax


def pass_through(input):
    return input


class JaxActivationNonLinearEnum(Enum):
    softplus = "softplus"
    tanh = jax.nn.tanh
    pass_through = pass_through


class JaxErrorTypeEnum(Enum):
    DFA = "DFA"
    DSEF = "DSEF"

class JaxMetaOptimizerEnum(Enum):
    Adam = "Adam"
    SGD = "SGD"

class JaxOptimizerModeEnum(Enum):
    BPTT = "BPTT"
    RTRL = "RTRL"
    SOFO = "SOFO"


class JaxRnnMetaLearnerOptions:
    def __init__(
        self,
        seed: int = 0,
        save_results: bool = True,
        results_subdir: str = "jax_rnn_meta_learner",
        metatrain_dataset: str = "emnist",
        display: bool = True,
        metaLearningRate: float = 0.001,
        numberOfClasses: int = 5,
        dataset_name: str = "emnist",
        chemicalInitialization="same",
        minTrainingDataPerClass: int = 5,
        maxTrainingDataPerClass: int = 10,
        queryDataPerClass: int = 10,
        input_size: int = 28,
        hidden_size: int = 128,
        output_size: int = 1,
        biological_min_tau: int = 5,
        biological_max_tau: int = 25,
        gradient: bool = False,
        outer_activation=JaxActivationNonLinearEnum.tanh,
        recurrent_activation=JaxActivationNonLinearEnum.softplus,
        number_of_time_steps: int = 10,
        load_model: str = None,
        load_optimizer: bool = False,
        dont_load_z_y: bool = False,
        error_type: JaxErrorTypeEnum = JaxErrorTypeEnum.DFA,
        low_dim_DFA: int = -1,
        permutation: bool = False,
        two_layer_RNN: bool = False,
        feedforward: bool = False,
        meta_optimizer: JaxMetaOptimizerEnum = JaxMetaOptimizerEnum.Adam,
        optimizer_mode: JaxOptimizerModeEnum = JaxOptimizerModeEnum.BPTT,
        sofo_samples: int = 60,
        sofo_damping: float = 1e-5,
        sofo_identity_sampling: bool = False,
        batch_size: int = 1,
        random_architectures: bool = False,
    ):
        self.seed = seed
        self.save_results = save_results
        self.results_subdir = results_subdir
        self.metatrain_dataset = metatrain_dataset
        self.display = display
        self.metaLearningRate = metaLearningRate
        self.numberOfClasses = numberOfClasses
        self.dataset_name = dataset_name
        self.chemicalInitialization = chemicalInitialization
        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau
        self.gradient = gradient
        self.outer_activation = outer_activation
        self.recurrent_activation = recurrent_activation
        self.number_of_time_steps = number_of_time_steps
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.dont_load_z_y = dont_load_z_y
        self.error_type = error_type
        self.low_dim_DFA = low_dim_DFA
        self.permutation = permutation
        self.two_layer_RNN = two_layer_RNN
        self.feedforward = feedforward
        self.meta_optimizer = meta_optimizer
        self.optimizer_mode = optimizer_mode
        self.sofo_samples = sofo_samples
        self.sofo_damping = sofo_damping
        self.sofo_identity_sampling = sofo_identity_sampling
        self.batch_size = batch_size
        self.random_architectures = random_architectures

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
