import datetime
from enum import Enum
from typing import Literal

#from synapses.benna_synapse import BennaSynapse
from synapses.complex_synapse import ComplexSynapse
#from synapses.GRU_synapse import GRUSynapse
#from synapses.individual_synapse import IndividualSynapse
#from synapses.LSTM_synapse import LSTMSynapse
#from synapses.reservoir_synapse import ReservoirSynapse


class schedulerEnum(Enum):
    exponential = "exponential"
    linear = "linear"
    constant = "constant"
    none = "none"


class optimizerEnum(Enum):
    adam = "adam"
    adamW = "adamW"
    sgd = "sgd"
    nadam = "nadam"
    radam = "radam"


class modelEnum(Enum):
    complex = ComplexSynapse
    #reservoir = ReservoirSynapse
    #individual = IndividualSynapse
    #benna = BennaSynapse
    #lstm = LSTMSynapse
    #gru = GRUSynapse


class chemicalEnum(Enum):
    same = "same"
    zero = "zero"
    different = "different"


class typeOfFeedbackEnum(Enum):
    FA = "FA"
    FA_NO_GRAD = "FA_NO_GRAD"
    DFA = "DFA"
    DFA_grad = "DFA_grad"
    DFA_grad_sign = "DFA_grad_sign"
    scalar = "scalar"
    DFA_grad_FA = "DFA_grad_FA"
    scalar_rate = "scalar_rate"
    scalar_rich = "scalar_rich"
    scalar_sign = "scalar_sign"
    scalar_minus_one = "scalar_minus_one"
    zero = "zero"
    target_propagation = "target_propagation"
    non_linear_DFA = "non_linear_DFA"


class sizeEnum(Enum):
    small = "small"
    normal = "normal"
    seven_layer = "seven_layer"
    nine_layer = "nine_layer"
    ten_layer = "ten_layer"
    six_layer = "six_layer"
    three_layer = "three_layer"
    three_layer_wide = "three_layer_wide"
    convolutional = "convolutional"


class MetaLearnerOptions:
    """
    Options for the metal learner
    """

    def __init__(
        self,
        scheduler: schedulerEnum,
        metaLossRegularization: int,
        biasLossRegularization: int,
        optimizer: optimizerEnum,
        model: modelEnum,
        results_subdir: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        seed: int = 0,
        size: sizeEnum = sizeEnum.normal,
        raytune: bool = False,
        save_results: bool = True,
        metatrain_dataset_1: str = None,
        metatrain_dataset_2: str = None,
        metatrain_dataset_3: str = None,
        display: bool = True,
        lr: float = 1e-3,
        numberOfClasses: int = 5,
        dataset_name: Literal["EMNIST", "FASHION-MNIST"] = "FASHION-MNIST",
        chemicalInitialization: chemicalEnum = chemicalEnum.same,
        trainSeparateFeedback: bool = False,
        feedbackSeparateModel: modelEnum = modelEnum.complex,
        trainSameFeedback: bool = False,
        minTrainingDataPerClass: int = 50,
        maxTrainingDataPerClass: int = 50,
        queryDataPerClass: int = 10,
        datasetDevice: Literal["cpu", "cuda"] = "cuda",
        continueTraining: str = None,
        typeOfFeedback: typeOfFeedbackEnum = typeOfFeedbackEnum.FA,
        dimOut: int = 47,
        loadModel: str = None,
        hrm_discount: int = 0,
        error_control: bool = False,
        leaky_error_alpha: float = 0,
        train_feedback_weights: bool = False,
        train_RCN: bool = False,
        wta: bool = False,
        shift_labels_2: int = 0,
        shift_labels_3: int = 0,
        scalar_variance_reduction: int = -1,
        low_rank_feedback: int = -1,
        split: bool = False,
        split_min_number_of_tasks: int = 2,
        split_max_number_of_tasks: int = 5,
        split_only_one_task_evaluation: int = -1,
        regenerate_feedback_weights: int = -1,
        no_training: bool = False,
    ):
        # print(trainSameFeedback)
        # assert (
        #    trainSameFeedback == True and trainSeperateFeedback== True
        # ), "Both trainSameFeedback and trainSeperateFeedback cannot be True"

        self.model = model
        self.size = size
        self.scheduler = scheduler
        self.metaLossRegularization = metaLossRegularization
        self.biasLossRegularization = biasLossRegularization
        self.optimizer = optimizer
        self.seed = seed
        self.raytune = raytune
        self.save_results = save_results
        self.results_subdir = results_subdir
        self.metatrain_dataset_1 = metatrain_dataset_1
        self.metatrain_dataset_2 = metatrain_dataset_2
        self.metatrain_dataset_3 = metatrain_dataset_3
        self.display = display
        self.lr = lr
        self.numberOfClasses = numberOfClasses
        self.dataset_name = dataset_name
        self.chemicalInitialization = chemicalInitialization
        self.trainSeparateFeedback = trainSeparateFeedback
        self.feedbackSeparateModel = feedbackSeparateModel
        self.trainSameFeedback = trainSameFeedback
        self.minTrainingDataPerClass = minTrainingDataPerClass
        self.maxTrainingDataPerClass = maxTrainingDataPerClass
        self.queryDataPerClass = queryDataPerClass
        self.datasetDevice = datasetDevice
        self.continueTraining = continueTraining
        self.typeOfFeedback = typeOfFeedback
        self.dimOut = dimOut
        self.loadModel = loadModel
        self.hrm_discount = hrm_discount
        self.error_control = error_control
        self.leaky_error_alpha = leaky_error_alpha
        self.train_feedback_weights = train_feedback_weights
        self.train_RCN = train_RCN
        self.wta = wta
        self.shift_labels_2 = shift_labels_2
        self.shift_labels_3 = shift_labels_3
        self.scalar_variance_reduction = scalar_variance_reduction
        self.low_rank_feedback = low_rank_feedback
        self.split = split
        self.split_min_number_of_tasks = split_min_number_of_tasks
        self.split_max_number_of_tasks = split_max_number_of_tasks
        self.split_only_one_task_evaluation = split_only_one_task_evaluation
        self.regenerate_feedback_weights = regenerate_feedback_weights
        self.no_training = no_training

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
