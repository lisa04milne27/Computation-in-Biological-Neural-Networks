import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from misc.dataset import (
    AddBernoulliTaskDataProcess,
    AddBernoulliTaskDataset,
    DataProcess,
    EmnistDataset,
    IMDBDataProcess,
    IMDBMetaDataset,
    IMDBWord2VecDataProcess,
    IMDBWord2VecMetaDataset,
)
from misc.utils import Plot, accuracy, log
from nn.jax_chemical_nn import JAXFeedforwardNN
from nn.jax_chemical_rnn import JAXChemicalRNN
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxErrorTypeEnum,
    JaxRnnMetaLearnerOptions,
)
from options.meta_learner_options import chemicalEnum
from synapses.jax_fast_rnn import JAXFastRnn
from torch.utils.data import DataLoader, RandomSampler


class JaxMetaLearnerRNN:

    def __init__(
        self,
        modelOptions,
        jaxMetaLearnerOptions,
        key: jax.random.key,
        numberOfChemicals: int = 1,
        metaTrainingDataset: Optional[DataLoader] = None,
    ):
        self.jaxMetaLearnerOptions = jaxMetaLearnerOptions
        key1, key2 = jax.random.split(key)
        self.key1 = key1
        self.key2 = key2
        if self.jaxMetaLearnerOptions.feedforward:
            self.rnn = JAXFeedforwardNN(
                dim_out=self.jaxMetaLearnerOptions.output_size,
                key=key1,
                input_size=self.jaxMetaLearnerOptions.input_size,
                gradient=self.jaxMetaLearnerOptions.gradient,
                activation=JaxActivationNonLinearEnum.softplus,
                error_type=self.jaxMetaLearnerOptions.error_type,
                low_dim_DFA=self.jaxMetaLearnerOptions.low_dim_DFA,
            )
        else:
            self.rnn = JAXChemicalRNN(
                input_size=self.jaxMetaLearnerOptions.input_size,
                key=key1,
                hidden_size=self.jaxMetaLearnerOptions.hidden_size,
                output_size=self.jaxMetaLearnerOptions.output_size,
                biological_min_tau=self.jaxMetaLearnerOptions.biological_min_tau,
                biological_max_tau=self.jaxMetaLearnerOptions.biological_max_tau,
                gradient=self.jaxMetaLearnerOptions.gradient,
                outer_activation=self.jaxMetaLearnerOptions.outer_activation,
                recurrent_activation=self.jaxMetaLearnerOptions.recurrent_activation,
                error_type=self.jaxMetaLearnerOptions.error_type,
                low_dim_DFA=self.jaxMetaLearnerOptions.low_dim_DFA,
                two_layer_RNN=self.jaxMetaLearnerOptions.two_layer_RNN,
            )
        self.save_results = self.jaxMetaLearnerOptions.save_results
        self.metaTrainingDataset = metaTrainingDataset
        if self.jaxMetaLearnerOptions.dataset_name == "ADDBERNOULLI":
            self.data_process = AddBernoulliTaskDataProcess(
                min_lag_1=5, max_lag_1=5, min_lag_2=8, max_lag_2=8, use_jax=True
            )
        elif self.jaxMetaLearnerOptions.dataset_name == "IMDB":
            self.data_process = IMDBDataProcess(
                minNumberOfSequencesPerClass=self.jaxMetaLearnerOptions.minTrainingDataPerClass,
                maxNumberOfSequencesPerClass=self.jaxMetaLearnerOptions.maxTrainingDataPerClass,
                use_jax=True,
            )
        elif self.jaxMetaLearnerOptions.dataset_name == "IMDB_WORD2VEC":
            self.data_process = IMDBWord2VecDataProcess(
                minNumberOfSequencesPerClass=self.jaxMetaLearnerOptions.minTrainingDataPerClass,
                maxNumberOfSequencesPerClass=self.jaxMetaLearnerOptions.maxTrainingDataPerClass,
                use_jax=True,
            )
        else:
            self.data_process = DataProcess(
                minTrainingDataPerClass=self.jaxMetaLearnerOptions.minTrainingDataPerClass,
                maxTrainingDataPerClass=self.jaxMetaLearnerOptions.maxTrainingDataPerClass,
                queryDataPerClass=self.jaxMetaLearnerOptions.queryDataPerClass,
                dimensionOfImage=28,
                iid=True,
                use_jax=True,
                permutation=self.jaxMetaLearnerOptions.permutation,
            )

        self.numberOfChemicals = numberOfChemicals
        self.metaOptimizer = JAXFastRnn(numberOfChemicals, modelOptions)
        # -- load model --
        self.metaOptimizer = eqx.tree_deserialise_leaves(
            self.jaxMetaLearnerOptions.load_model + "/meta_learner_model.eqx", self.metaOptimizer
        )

        # -- loss function --
        self.loss_function = optax.safe_softmax_cross_entropy

        # -- log params
        self.result_directory = os.getcwd() + "/results_4"
        if self.save_results:
            self.result_directory = os.getcwd() + "/results_4"
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + self.jaxMetaLearnerOptions.results_subdir
                + "/"
                + str(self.jaxMetaLearnerOptions.minTrainingDataPerClass)
            )
            os.makedirs(self.result_directory, exist_ok=False)
            with open(self.result_directory + "/arguments.txt", "w") as f:
                f.writelines(str(self.jaxMetaLearnerOptions))
                f.writelines(str(modelOptions))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)

    def chemicals_init(self):
        layers = self.rnn.layers
        self.synaptic_weights = []
        for i in layers:
            # self.synaptic_weights.append(jnp.empty((numberOfChemicals, i.in_features, i.out_features)))
            ## Initialize to xavier initialization
            if self.jaxMetaLearnerOptions.chemicalInitialization == chemicalEnum.same:
                temp = jax.nn.initializers.xavier_uniform()(self.key2, (i.in_features, i.out_features))
                holder = jnp.tile(temp, (self.numberOfChemicals, 1, 1))
            elif self.jaxMetaLearnerOptions.chemicalInitialization == chemicalEnum.different:
                holder = jax.nn.initializers.xavier_uniform()(
                    self.key2, (self.numberOfChemicals, i.in_features, i.out_features)
                )
            self.synaptic_weights.append(holder)
        self.synaptic_weights = tuple(self.synaptic_weights)

    def inner_loop_per_image(self, carry, input):
        synaptic_weights, parameters, rnn, hidden_state, metaOptimizer = carry
        x, labels = input

        if self.jaxMetaLearnerOptions.dataset_name != "ADDBERNOULLI":
            labels = jax.nn.one_hot(labels, num_classes=self.jaxMetaLearnerOptions.output_size)

        y, hidden_state, activations_arr, errors_arr = rnn(x, hidden_state, labels)

        def update_layer(w, p, act, err):
            return metaOptimizer(w, p, act, err)

        activations_arr = tuple(activations_arr)
        errors_arr = tuple(errors_arr)

        results = jax.tree_util.tree_map(update_layer, synaptic_weights, parameters, activations_arr, errors_arr)
        new_parameters = tuple(res[0] for res in results)
        new_synaptic_weights = tuple(res[1] for res in results)

        num_layers = len(new_parameters)

        new_rnn = eqx.tree_at(
            lambda r: (r.layers, *(getattr(r, f"forward{i+1}") for i in range(num_layers))),
            rnn,
            (new_parameters, *new_parameters),
        )

        return (
            new_synaptic_weights,
            new_parameters,
            new_rnn,
            hidden_state,
            metaOptimizer,
        ), y

    def full_inner_loop(self, carry, input) -> jnp.ndarray:
        synaptic_weights, parameters, rnn, metaOptimizer = carry
        hidden_state = rnn.initialise_hidden_state(batch_size=1)
        x, label = input

        if self.jaxMetaLearnerOptions.dataset_name == "ADDBERNOULLI":
            x = jnp.reshape(x, (x.shape[0], -1))  # (time_steps, input_size)
        elif (
            self.jaxMetaLearnerOptions.dataset_name == "IMDB"
            or self.jaxMetaLearnerOptions.dataset_name == "IMDB_WORD2VEC"
        ):
            pass  # x is already in shape (time_steps, input_size)
        else:
            x = jnp.reshape(x, (self.jaxMetaLearnerOptions.number_of_time_steps, -1))  # (time_steps, input_size)

        if self.jaxMetaLearnerOptions.dataset_name == "ADDBERNOULLI":
            label = jnp.reshape(label, (label.shape[0], -1))  # (time_steps, output_size)
        elif (
            self.jaxMetaLearnerOptions.dataset_name == "IMDB"
            or self.jaxMetaLearnerOptions.dataset_name == "IMDB_WORD2VEC"
        ):
            label = jnp.repeat(label, repeats=x.shape[0], axis=0)
        else:
            label = jnp.repeat(
                label, repeats=self.jaxMetaLearnerOptions.number_of_time_steps, axis=0
            )  # (time_steps, output_size)

        (new_synaptic_weights, new_parameters, new_rnn, hidden_state, metaOptimizer), y = jax.lax.scan(
            self.inner_loop_per_image,
            (synaptic_weights, parameters, rnn, hidden_state, metaOptimizer),
            (x, label),
        )
        return (new_synaptic_weights, new_parameters, new_rnn, metaOptimizer), (y, hidden_state)

    def inner_loop_per_image_no_update(self, carry, input):
        hidden_state, rnn = carry
        x = input
        y, hidden_state, _, _ = rnn(x, hidden_state, None)
        return (hidden_state, rnn), y

    def full_inner_loop_no_update(self, input) -> jnp.ndarray:
        x, number_of_timesteps, rnn, current_hidden_state = input

        if self.jaxMetaLearnerOptions.dataset_name == "ADDBERNOULLI":
            hidden_state = current_hidden_state
            hidden_state = jnp.reshape(hidden_state, (1, hidden_state.shape[-1]))
        else:
            hidden_state = rnn.initialise_hidden_state(batch_size=x.shape[0])

        if self.jaxMetaLearnerOptions.dataset_name == "ADDBERNOULLI":
            x = jnp.reshape(x, (x.shape[0], x.shape[1], -1))  # (batch_size, time_steps, input_size)
        elif (
            self.jaxMetaLearnerOptions.dataset_name == "IMDB"
            or self.jaxMetaLearnerOptions.dataset_name == "IMDB_WORD2VEC"
        ):
            pass  # x is already in shape (batch_size, time_steps, input_size)
        else:
            x = jnp.reshape(x, (x.shape[0], number_of_timesteps, -1))  # (batch_size, time_steps, input_size)

        # -- use vmap to process all sequences in the batch --
        def run_sequence(hidden_state, rnn, x):
            return jax.lax.scan(
                self.inner_loop_per_image_no_update,
                (hidden_state, rnn),
                (x),
            )

        (_, _), y = jax.vmap(
            run_sequence,
            in_axes=(0, None, 0),
        )(hidden_state, rnn, x)
        return y

    # @eqx.filter_jit
    def compute_meta_loss(
        self,
        metaOptimizer,
        synaptic_weights,
        rnn_layers,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
    ):

        synaptic_weights_tuple = synaptic_weights
        parameters_tuple = rnn_layers
        rnn = rnn

        # -- inner loop --
        carry_init = (
            synaptic_weights_tuple,
            parameters_tuple,
            rnn,
            metaOptimizer,
        )

        (new_synaptic_weights, new_parameters, new_rnn, _), (y, current_hidden_state) = jax.lax.scan(
            self.full_inner_loop,
            carry_init,
            (x_trn, y_trn),
        )

        qry_predictions = self.full_inner_loop_no_update(
            (x_qry, self.jaxMetaLearnerOptions.number_of_time_steps, new_rnn, current_hidden_state)
        )

        if self.jaxMetaLearnerOptions.dataset_name == "ADDBERNOULLI":
            qry_predictions = jnp.squeeze(qry_predictions)
            y_qry = jnp.squeeze(y_qry)
            losses = self.loss_function(qry_predictions, y_qry)
            avg_loss = jnp.mean(losses)
            acc = -1  # accuracy not defined for this task
        else:
            qry_predictions = qry_predictions[:, -1, :]  # take last time step
            one_hot_targets = jax.nn.one_hot(y_qry.flatten(), num_classes=self.jaxMetaLearnerOptions.output_size)
            losses = self.loss_function(qry_predictions, one_hot_targets)
            avg_loss = jnp.mean(losses)
            acc = accuracy(qry_predictions, y_qry.ravel(), use_jax=True)

        return avg_loss, acc

    @eqx.filter_jit
    def make_step(
        self,
        metaOptimizer,
        synaptic_weights_tuple,
        rnn_layers,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
    ):

        loss, acc = self.compute_meta_loss(
            metaOptimizer, synaptic_weights_tuple, rnn_layers, rnn, x_trn, y_trn, x_qry, y_qry
        )

        return loss, acc

    def train(self):
        current_training_data_per_class = None
        for eps, data in enumerate(self.metaTrainingDataset):

            if self.jaxMetaLearnerOptions.dataset_name == "ADDBERNOULLI":
                x_trn, y_trn, x_qry, y_qry, roll_1, roll_2 = self.data_process(
                    data
                )  # current_training_data is current lag
                x_trn = jnp.expand_dims(x_trn, 0)
                y_trn = jnp.expand_dims(y_trn, 0)
                x_qry = jnp.expand_dims(x_qry, 0)
                y_qry = jnp.expand_dims(y_qry, 0)

                current_training_data_per_class = x_trn.shape[1]
            elif (
                self.jaxMetaLearnerOptions.dataset_name == "IMDB"
                or self.jaxMetaLearnerOptions.dataset_name == "IMDB_WORD2VEC"
            ):
                x_trn, y_trn, x_qry, y_qry, current_training_data_per_class = self.data_process(data)
            else:
                x_trn, y_trn, x_qry, y_qry, current_training_data_per_class, _ = self.data_process(
                    data, self.jaxMetaLearnerOptions.numberOfClasses
                )
            # -- convert to jax arrays --
            x_trn = jnp.array(x_trn)
            y_trn = jnp.array(y_trn)
            x_qry = jnp.array(x_qry)
            y_qry = jnp.array(y_qry)

            # -- weight initialization --
            self.rnn = self.rnn.reset_weights(self.key2)

            # -- initialize chemicals --
            self.chemicals_init()
            new_parameters = list(self.rnn.layers)
            new_synaptic_weights = list(self.synaptic_weights)
            for idx, parameter in enumerate(self.rnn.layers):
                synaptic_weight = self.synaptic_weights[idx]
                new_synaptic_weight, new_parameter = self.metaOptimizer.initialize_parameters(
                    synaptic_weight, parameter
                )
                new_synaptic_weights[idx] = new_synaptic_weight
                new_parameters[idx] = new_parameter
            self.synaptic_weights = tuple(new_synaptic_weights)
            self.new_parameters = tuple(new_parameters)
            num_layers = len(self.new_parameters)
            self.rnn = eqx.tree_at(
                lambda r: (r.layers, *(getattr(r, f"forward{i+1}") for i in range(num_layers))),
                self.rnn,
                (self.new_parameters, *self.new_parameters),
            )
            # -- meta-optimization --
            loss, acc = self.make_step(
                self.metaOptimizer,
                self.synaptic_weights,
                self.rnn.layers,
                self.rnn,
                x_trn,
                y_trn,
                x_qry,
                y_qry,
            )

            if self.save_results:
                log([loss.item()], self.result_directory + "/loss_meta.txt")
                log([acc], self.result_directory + "/acc_meta.txt")

            line = "Runner Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}\tTraining/Class: {}".format(
                eps + 1, loss.item(), acc, current_training_data_per_class
            )
            if self.jaxMetaLearnerOptions.display:
                print(line)

            if self.save_results:

                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

        if self.save_results:
            self.plot()
            # -- save model --
            eqx.tree_serialise_leaves(self.result_directory + "/meta_learner_model.eqx", self.metaOptimizer)

        print("Training completed.")


def jax_runner(index: int, result_subdirectory: str, modelPath: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # second gpu
    key = jax.random.PRNGKey(42)
    # jax.config.update("jax_enable_x64", False)

    # -- load data
    numWorkers = 2
    training_data = [20, 30, 40, 50, 80, 150, 200, 225, 375]
    """training_data = [
        # 10,
        # 9,
        10,
        20,
        30,
        # 50,
        # 75,
        # 90,
        # 30,
        # 100,
        # 200,
        # 300,
        # 500,
        # 700,
        # 1000,
        # 2000,
        # 3000,
        # 4000,
        # 6000,
        # 8000,
        40,
        50,
        60,
        70,
        80,
        90,
        # 100,
        # 110,
        # 120,
        # 130,
        # 10000,
        # 12000,
        # 14000,
        # 15000,
    ]"""
    if index >= len(training_data):
        return
    epochs = 20

    dataset_name = "EMNIST"
    minTrainingDataPerClass = training_data[index]
    maxTrainingDataPerClass = training_data[index]
    queryDataPerClass = 20
    numberOfTimeSteps = 1

    if dataset_name == "EMNIST":
        numberOfClasses = 5
        dataset = EmnistDataset(
            minTrainingDataPerClass=minTrainingDataPerClass,
            maxTrainingDataPerClass=maxTrainingDataPerClass,
            queryDataPerClass=queryDataPerClass,
            dimensionOfImage=28,
            use_jax=True,
        )
        dimOut = 47
        dimIn = int(28 * 28 / numberOfTimeSteps)
    elif dataset_name == "ADDBERNOULLI":
        queryDataPerClass = 100
        dataset = AddBernoulliTaskDataset(
            minSequenceLength=minTrainingDataPerClass,
            maxSequenceLength=maxTrainingDataPerClass,
            querySequenceLength=100,
        )
        dimOut = 2
        dimIn = 2
        numberOfClasses = 1
    elif dataset_name == "IMDB":
        numberOfClasses = 2
        dimOut = 2
        queryDataPerClass = 20
        dataset = IMDBMetaDataset(
            minNumberOfSequences=minTrainingDataPerClass,
            maxNumberOfSequences=maxTrainingDataPerClass,
            query_q=queryDataPerClass,
            max_seq_len=200,
        )
        numWorkers = 0
        dimIn = 768
    elif dataset_name == "IMDB_WORD2VEC":
        numberOfClasses = 2
        dimOut = 2
        queryDataPerClass = 20
        dataset = IMDBWord2VecMetaDataset(
            minNumberOfSequences=minTrainingDataPerClass,
            maxNumberOfSequences=maxTrainingDataPerClass,
            query_q=queryDataPerClass,
            max_seq_len=200,
        )
        numWorkers = 0
        dimIn = 300

    sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses)
    metatrain_dataset = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=numberOfClasses,
        drop_last=True,
        num_workers=numWorkers,
        persistent_workers=False,
    )

    # -- options
    modelOptions = None
    modelOptions = fastRnnOptions(
        nonLinear=JaxActivationNonLinearEnum.tanh,
        update_rules=[0, 1, 2, 3, 4, 9, 12],
        minSlowTau=2,
        maxSlowTau=50,
        y_vector=yVectorEnum.none,
        z_vector=zVectorEnum.default,
        operator=operatorEnum.mode_9,
    )
    # cuda:1
    # device = "cpu"
    current_dir = os.getcwd()
    # runner = current_dir + "/results_3/jax_rnn_12/20260121-024411"
    runner = (
        current_dir + "/results_4/Jax_rnn_fixed_28/20260603-160027"
    )  # mode_9_rand_converted"#Jax_9_chem_scalar/20260526-030351"
    # -- meta-learner options
    metaLearnerOptions = JaxRnnMetaLearnerOptions(
        seed=42,
        save_results=True,
        results_subdir=result_subdirectory,
        metatrain_dataset=dataset_name,
        display=True,
        metaLearningRate=None,
        numberOfClasses=numberOfClasses,
        dataset_name=dataset_name,
        chemicalInitialization=chemicalEnum.different,
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        input_size=dimIn,
        hidden_size=128,
        output_size=dimOut,
        biological_min_tau=1,
        biological_max_tau=28,
        gradient=True,
        outer_activation=JaxActivationNonLinearEnum.tanh,
        recurrent_activation=JaxActivationNonLinearEnum.softplus,
        number_of_time_steps=numberOfTimeSteps,
        load_model=modelPath,
        error_type=JaxErrorTypeEnum.DSEF,
        low_dim_DFA=-1,
        permutation=False,
        two_layer_RNN=False,
        feedforward=True,
    )

    metalearning_model = JaxMetaLearnerRNN(
        modelOptions=modelOptions,
        jaxMetaLearnerOptions=metaLearnerOptions,
        key=key,
        numberOfChemicals=13,
        metaTrainingDataset=metatrain_dataset,
    )

    metalearning_model.train()


def main_jax_runner():
    outer = os.getcwd() + "/results_4/Jax_13_chem_DSEF_full_sweep_200"
    tau_min = os.listdir(outer)
    for tau in tau_min:
        inner = outer + "/" + tau
        tau_max = os.listdir(inner)
        for tau2 in tau_max:
            inner2 = inner + "/" + tau2
            for index_outer in range(0, 30):
                jax_runner(
                    index=index_outer,
                    result_subdirectory="runner_13_chem_DSEF_full_sweep_200_lisa/" + tau + "/" + tau2,
                    modelPath=inner2,
                )
