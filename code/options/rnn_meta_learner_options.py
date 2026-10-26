import datetime
from enum import Enum
from typing import Literal

import torch.nn.functional as functional
from options.meta_learner_options import chemicalEnum, optimizerEnum
from synapses.complex_rnn import ComplexRnn
#from synapses.fast_rnn import FastRnn
#from synapses.kernel_rnn import KernelRnn


def pass_through(input):
    return input


class activationNonLinearEnum(Enum):
    relu = functional.relu
    softplus = "softplus"
    tanh = functional.tanh
    pass_through = pass_through


class rnnModelEnum(Enum):
    """
    Enum for the model
    """

    complex = ComplexRnn
    #kernel = KernelRnn
    #fast = FastRnn


class errorEnum(Enum):
    """
    Enum for the error function
    """

    last = "last"
    all = "all"


class recurrentInitEnum(Enum):
    """
    Enum for the recurrent initialization
    """

    identity = "identity"
    xavierUniform = "xavierUniform"


class RnnMetaLearnerOptions:
    """
    Options for the metal learner
    """

    def __init__(
        self,
        optimizer: optimizerEnum,
        model: rnnModelEnum,
        results_subdir: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        seed: int = 0,
        save_results: bool = True,
        metatrain_dataset: str = None,
        display: bool = True,
        lr: float = 1e-3,
        numberOfClasses: int = 5,
        dataset_name: Literal["EMNIST", "FASHION-MNIST"] = "EMNIST",
        chemicalInitialization: chemicalEnum = chemicalEnum.same,
        minTrainingDataPerClass: int = 50,
        maxTrainingDataPerClass: int = 50,
        queryDataPerClass: int = 10,
        rnn_input_size: int = 28,
        datasetDevice: Literal["cpu", "cuda"] = "cuda",
        continueTraining: str = None,
        reset_fast_weights: bool = True,
        requireFastChemical: bool = False,
        slowIsFast: bool = False,
        dimOut: int = 1,
        biological: bool = False,
        biological_min_tau: int = 1,
        biological_max_tau: int = 56,
        error=errorEnum.last,
        leaky_error: float = 0.0,
        hidden_reset: bool = True,
        loss_meta_logits_all: bool = False,
        hidden_size: int = 128,
        recurrent_init: recurrentInitEnum = recurrentInitEnum.xavierUniform,
        test_time_training: bool = False,
        diff_hidden_error: bool = False,
        gradient: bool = False,
        easy_gradient: bool = False,
        hrm_discount: int = 200,
        outer_non_linear=activationNonLinearEnum.tanh,
        recurrent_non_linear=activationNonLinearEnum.pass_through,
        post_recurrent_non_linear=activationNonLinearEnum.tanh,
    ):

        self.model = model
        self.optimizer = optimizer
        self.seed = seed
        self.save_results = save_results
        self.results_subdir = results_subdir
        self.metatrain_dataset = metatrain_dataset
        self.display = display
        self.lr = lr
        self.numberOfClasses = numberOfClasses
        self.dataset_name = dataset_name
        self.chemicalInitialization = chemicalInitialization
        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.rnn_input_size = rnn_input_size
        self.queryDataPerClass = queryDataPerClass
        self.datasetDevice = datasetDevice
        self.continueTraining = continueTraining
        self.reset_fast_weights = reset_fast_weights
        self.requireFastChemical = requireFastChemical
        self.slowIsFast = slowIsFast
        self.dimOut = dimOut
        self.biological = biological
        self.biological_min_tau = biological_min_tau
        self.biological_max_tau = biological_max_tau
        self.error = error
        self.leaky_error = leaky_error
        self.hidden_reset = hidden_reset
        self.loss_meta_logits_all = loss_meta_logits_all
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.test_time_training = test_time_training
        self.diff_hidden_error = diff_hidden_error
        self.gradient = gradient
        self.easy_gradient = easy_gradient
        self.hrm_discount = hrm_discount
        self.outer_non_linear = outer_non_linear
        self.recurrent_non_linear = recurrent_non_linear
        self.post_recurrent_non_linear = post_recurrent_non_linear

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
