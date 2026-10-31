import os
from pathlib import Path
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from misc.jax_data import DataProcess, EmnistDataset, emnist_batches
from misc.jax_utils import Plot, accuracy, log
from nn.jax_chemical_nn import JAXFeedforwardNN
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxErrorTypeEnum,
    JaxRnnMetaLearnerOptions,
)
from synapses.jax_fast_rnn import JAXFastRnn

DATASET_NAME = "emnist"


class JaxMetaLearnerRNN:

    def __init__(
        self,
        modelOptions,
        jaxMetaLearnerOptions,
        key: jax.random.key,
        numberOfChemicals: int = 1,
        metaTrainingDataset: Optional[EmnistDataset] = None,
        #Changing this below
        tracked_layer: int = 3,
        tracked_neuron: int = 5,
        tracked_input_synapse: int = 0,
    ):
        self.jaxMetaLearnerOptions = jaxMetaLearnerOptions
        key1, key2 = jax.random.split(key)
        self.key1 = key1
        self.key2 = key2
        self.rnn = JAXFeedforwardNN(
            dim_out=self.jaxMetaLearnerOptions.output_size,
            key=key1,
            input_size=self.jaxMetaLearnerOptions.input_size,
            gradient=self.jaxMetaLearnerOptions.gradient,
            activation=JaxActivationNonLinearEnum.softplus,
            error_type=self.jaxMetaLearnerOptions.error_type,
            low_dim_DFA=self.jaxMetaLearnerOptions.low_dim_DFA,
        )
        self.save_results = self.jaxMetaLearnerOptions.save_results
        self.metaTrainingDataset = metaTrainingDataset
        # Zero-based indexes into the forward layers and their output neurons.
        self.tracked_layer = tracked_layer
        self.tracked_neuron = tracked_neuron
        self.tracked_input_synapse = tracked_input_synapse
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
        self.result_directory = str(Path(__file__).resolve().parents[2] / "results_4")
        if self.save_results:
            os.makedirs(self.result_directory, exist_ok=True)
            self.result_directory += (
                "/"
                + self.jaxMetaLearnerOptions.results_subdir
                + "/"
                + str(self.jaxMetaLearnerOptions.minTrainingDataPerClass)
            )
            # Reuse an existing run directory so rerunning the documented
            # entry point does not fail before training starts.
            os.makedirs(self.result_directory, exist_ok=True)
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
            if self.jaxMetaLearnerOptions.chemicalInitialization == "same":
                temp = jax.nn.initializers.xavier_uniform()(self.key2, (i.in_features, i.out_features))
                holder = jnp.tile(temp, (self.numberOfChemicals, 1, 1))
            elif self.jaxMetaLearnerOptions.chemicalInitialization == "different":
                holder = jax.nn.initializers.xavier_uniform()(
                    self.key2, (self.numberOfChemicals, i.in_features, i.out_features)
                )
            self.synaptic_weights.append(holder)
        self.synaptic_weights = tuple(self.synaptic_weights)

    def inner_loop_per_image(self, carry, input):
        synaptic_weights, parameters, rnn, hidden_state, metaOptimizer = carry
        x, labels = input

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

        # Synaptic state shape is (chemicals, input_synapses, output_neurons).
        # Save all seven chemical values for every incoming synapse to the
        # selected neuron, immediately after this training image's update.
        tracked_chemicals = new_synaptic_weights[self.tracked_layer][
            :, :, self.tracked_neuron
        ].T

        return (
            new_synaptic_weights,
            new_parameters,
            new_rnn,
            hidden_state,
            metaOptimizer,
        ), (y, tracked_chemicals)

    def full_inner_loop(self, carry, input) -> jnp.ndarray:
        synaptic_weights, parameters, rnn, metaOptimizer = carry
        x, label = input

        number_of_time_steps = self.jaxMetaLearnerOptions.number_of_time_steps
        # The outer scan item is one complete task (all examples in the
        # task), not a single image. Flatten the task/example axes while
        # preserving the image feature dimension, then scan over examples.
        x = jnp.reshape(x, (-1, number_of_time_steps, self.jaxMetaLearnerOptions.input_size))
        label = jnp.reshape(label, (-1,))
        x = jnp.reshape(x, (-1, self.jaxMetaLearnerOptions.input_size))
        # Labels are class indices for each example. Repeat each label only
        # when an image itself is split into multiple recurrent time steps.
        label = jnp.repeat(label, number_of_time_steps)

        (new_synaptic_weights, new_parameters, new_rnn, hidden_state, metaOptimizer), (y, tracked_chemicals) = jax.lax.scan(
            self.inner_loop_per_image,
            (synaptic_weights, parameters, rnn, None, metaOptimizer),
            (x, label),
        )
    
        return (new_synaptic_weights, new_parameters, new_rnn, metaOptimizer), (y, hidden_state, tracked_chemicals)

    def inner_loop_per_image_no_update(self, carry, input):
        hidden_state, rnn = carry
        x = input
        y, hidden_state, _, _ = rnn(x, hidden_state, None)
        return (hidden_state, rnn), y

    def full_inner_loop_no_update(self, x, number_of_timesteps, rnn):
        # DataProcess supplies (tasks, examples, flattened_pixels). Match the
        # training path: partition pixels into recurrent steps only when the
        # configured step count requires it. With one step, each input remains
        # the full 784-feature image expected by forward1.
        input_size = self.jaxMetaLearnerOptions.input_size
        x = jnp.reshape(x, (x.shape[0], x.shape[1], number_of_timesteps, input_size))
        predict = lambda sequence: jax.vmap(
            lambda steps: jax.vmap(lambda step: rnn(step, None, None)[0])(steps)
        )(sequence)
        return jax.vmap(predict)(x)

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

        (new_synaptic_weights, new_parameters, new_rnn, _), (y, current_hidden_state, tracked_chemicals) = jax.lax.scan(
            self.full_inner_loop,
            carry_init,
            (x_trn, y_trn),
        )

        qry_predictions = self.full_inner_loop_no_update(
            x_qry, self.jaxMetaLearnerOptions.number_of_time_steps, new_rnn
        )

        # Shape is (tasks, examples, time_steps, classes). Select the final
        # recurrent output, then flatten task/example axes to match y_qry.
        qry_predictions = qry_predictions[:, :, -1, :]
        qry_predictions = qry_predictions.reshape((-1, self.jaxMetaLearnerOptions.output_size))
        query_labels = y_qry.reshape((-1,))
        one_hot_targets = jax.nn.one_hot(query_labels, num_classes=self.jaxMetaLearnerOptions.output_size)
        losses = self.loss_function(qry_predictions, one_hot_targets)
        avg_loss = jnp.mean(losses)
        acc = accuracy(qry_predictions, query_labels, use_jax=True)

        return avg_loss, acc, tracked_chemicals

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

        loss, acc, tracked_chemicals = self.compute_meta_loss(
            metaOptimizer, synaptic_weights_tuple, rnn_layers, rnn, x_trn, y_trn, x_qry, y_qry
        )

        return loss, acc, tracked_chemicals


    def train(self):
       current_training_data_per_class = None
    
       for eps, data in enumerate(self.metaTrainingDataset):

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
           loss, acc, tracked_chemicals = self.make_step(
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
               self.save_neuron_chemical_plot(tracked_chemicals, eps + 1)

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

    def save_neuron_chemical_plot(self, chemical_history, episode: int):
        """Plot all seven chemical values for one synapse across images."""
        history = np.asarray(chemical_history)
        # One task is the expected configuration. If several are present,
        # plot the first task to keep the x-axis as its ordered images.
        if history.ndim == 4:
            history = history[0]
        # Expected shape: (training_images, incoming_synapses, chemicals).
        if history.ndim != 3 or history.shape[-1] != self.numberOfChemicals:
            raise ValueError(f"Expected (images, synapses, chemicals), got {history.shape}")
        if not 0 <= self.tracked_input_synapse < history.shape[1]:
            raise ValueError(
                f"Incoming synapse {self.tracked_input_synapse} is out of range "
                f"for layer {self.tracked_layer + 1} with {history.shape[1]} inputs"
            )

        selected_synapse = history[:, self.tracked_input_synapse, :]
        csv_path = os.path.join(
            self.result_directory,
            f"neuron_synapse_chemicals_episode_{episode:04d}.csv",
        )
        with open(csv_path, "w", newline="") as csv_file:
            csv_file.write("training_image,incoming_synapse,chemical,_value\n")
            for image_index in range(history.shape[0]):
                for synapse_index in range(history.shape[1]):
                    for chemical_index in range(history.shape[2]):
                        csv_file.write(
                            f"{image_index + 1},{synapse_index + 1},{chemical_index + 1},"
                            f"{history[image_index, synapse_index, chemical_index]:.10g}\n"
                        )
        images = np.arange(1, selected_synapse.shape[0] + 1)
        fig, ax = plt.subplots(figsize=(11, 6))
        for chemical in range(selected_synapse.shape[1]):
            ax.plot(images, selected_synapse[:, chemical], label=f"Chemical {chemical + 1}")
        ax.set_title(
            f"Layer {self.tracked_layer + 1}, neuron {self.tracked_neuron + 1}, "
            f"incoming synapse {self.tracked_input_synapse + 1}: chemical values "
            f"after each training image (episode {episode})"
        )
        ax.set_xlabel("Training image")
        ax.set_ylabel("Synaptic chemical value")
        ax.set_xlim(1, len(images))
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.result_directory, f"neuron_synapse_chemicals_episode_{episode:04d}.png"),
            dpi=150,
        )
        plt.close(fig)


import matplotlib.pyplot as plt
import numpy as np



def jax_runner(index: int, result_subdirectory: str, modelPath: str):
    key = jax.random.PRNGKey(42)
    # jax.config.update("jax_enable_x64", False)

    # -- load data
    training_data = [20, 30, 40, 50, 80, 150, 200, 225, 375]
    if index >= len(training_data):
        return
    epochs = 20

    #minTrainingDataPerClass = training_data[index]
    #maxTrainingDataPerClass = training_data[index]
    minTrainingDataPerClass = 70
    maxTrainingDataPerClass = 70
    queryDataPerClass = 20
    numberOfTimeSteps = 1

    numberOfClasses = 5
    dataset = EmnistDataset(
        minTrainingDataPerClass=minTrainingDataPerClass,
        maxTrainingDataPerClass=maxTrainingDataPerClass,
        queryDataPerClass=queryDataPerClass,
        dimensionOfImage=28,
    )
    dimOut = 47
    dimIn = int(28 * 28 / numberOfTimeSteps)

    metatrain_dataset = emnist_batches(dataset, epochs, numberOfClasses, numberOfClasses)

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
    # -- meta-learner options
    metaLearnerOptions = JaxRnnMetaLearnerOptions(
        seed=42,
        save_results=True,
        results_subdir=result_subdirectory,
        metatrain_dataset=DATASET_NAME,
        display=True,
        metaLearningRate=None,
        numberOfClasses=numberOfClasses,
        dataset_name=DATASET_NAME,
        chemicalInitialization="different",
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
    root = Path(__file__).resolve().parents[2] / "results_4" / "Jax_13_chem_DSEF_full_sweep_200"
    # Run the selected inner loop with tau_min=1, tau_max=400 and 20 examples per class.
    model_path = str(root / "1" / "400")
    jax_runner(
        index=0,
        result_subdirectory="inner_loop",
        modelPath=model_path,
    )
