import datetime
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
#from misc.sofo_api import ggn_ce, jmp, sample_v
from misc.utils import Plot, accuracy, log
from nn.jax_chemical_nn import JAXFeedforwardNN
from nn.jax_chemical_rnn import JAXChemicalRNN
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxErrorTypeEnum,
    JaxMetaOptimizerEnum,
    JaxOptimizerModeEnum,
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
        key1, key2, key3 = jax.random.split(key, num=3)
        rnn_keys = jax.random.split(key1, num=self.jaxMetaLearnerOptions.batch_size)
        self.key2_split = jax.random.split(key2, num=self.jaxMetaLearnerOptions.batch_size)
        self.key3 = key3

        if self.jaxMetaLearnerOptions.random_architectures:
            self.rnn_rnns = jax.vmap(lambda k: JAXChemicalRNN(
                input_size=self.jaxMetaLearnerOptions.input_size,
                key=k,
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
            ))(rnn_keys)
            self.rnn_ffs =  jax.vmap(lambda k: JAXFeedforwardNN(
                dim_out=self.jaxMetaLearnerOptions.output_size,
                key=k,
                input_size=784,
                gradient=self.jaxMetaLearnerOptions.gradient,
                activation=JaxActivationNonLinearEnum.softplus,
                error_type=self.jaxMetaLearnerOptions.error_type,
                low_dim_DFA=self.jaxMetaLearnerOptions.low_dim_DFA,
            ))(rnn_keys)
            self.true_number_of_timesteps = self.jaxMetaLearnerOptions.number_of_time_steps
        elif not self.jaxMetaLearnerOptions.feedforward:
            self.rnns = jax.vmap(lambda k: JAXChemicalRNN(
                input_size=self.jaxMetaLearnerOptions.input_size,
                key=k,
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
            ))(rnn_keys)
        else:
            self.rnns =  jax.vmap(lambda k: JAXFeedforwardNN(
                dim_out=self.jaxMetaLearnerOptions.output_size,
                key=k,
                input_size=784,
                gradient=self.jaxMetaLearnerOptions.gradient,
                activation=JaxActivationNonLinearEnum.softplus,
                error_type=self.jaxMetaLearnerOptions.error_type,
                low_dim_DFA=self.jaxMetaLearnerOptions.low_dim_DFA,
            ))(rnn_keys)
            
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
                batch_size=self.jaxMetaLearnerOptions.batch_size,
            )

        self.numberOfChemicals = numberOfChemicals
        self.metaOptimizer = JAXFastRnn(numberOfChemicals, modelOptions)
        # -- load model if specified --
        if self.jaxMetaLearnerOptions.load_model is not None:
            if self.jaxMetaLearnerOptions.dont_load_z_y:
                # load the model but keep v and y vectors as they are
                loaded_metaOptimizer = eqx.tree_deserialise_leaves(
                    self.jaxMetaLearnerOptions.load_model + "/meta_learner_model.eqx", self.metaOptimizer
                )
                self.metaOptimizer = eqx.tree_at(
                    lambda m: (m.Q_matrix, m.K_matrix, m.v_vector),
                    self.metaOptimizer,
                    (
                        loaded_metaOptimizer.Q_matrix,
                        loaded_metaOptimizer.K_matrix,
                        loaded_metaOptimizer.v_vector,
                    ),
                )
            else:
                self.metaOptimizer = eqx.tree_deserialise_leaves(
                    self.jaxMetaLearnerOptions.load_model + "/meta_learner_model.eqx", self.metaOptimizer
                )

        # -- optimizer --
        trainable_mask = self.get_trainable_mask(self.metaOptimizer)
        if self.jaxMetaLearnerOptions.meta_optimizer == JaxMetaOptimizerEnum.Adam:
            self.optimizer = optax.chain(
                #optax.clip(1.0), #clip_by_global_norm(1.0),  # Max norm of 1.0
                optax.adam(learning_rate=self.jaxMetaLearnerOptions.metaLearningRate),
            )
        elif self.jaxMetaLearnerOptions.meta_optimizer == JaxMetaOptimizerEnum.SGD:
            constant_sched = optax.constant_schedule(self.jaxMetaLearnerOptions.metaLearningRate)
            decay_sched = optax.exponential_decay(
                init_value=self.jaxMetaLearnerOptions.metaLearningRate,
                transition_steps=100,
                decay_rate=0.95,
                staircase=True,
                end_value=self.jaxMetaLearnerOptions.metaLearningRate * 0.01 # Floor it at 1% of the initial LR
            )
            self.scheduler = optax.join_schedules(
                schedules=[constant_sched, decay_sched],
                boundaries=[1500]
            )
            self.optimizer = optax.chain(
                #optax.clip(1.0), #_by_global_norm(1.0),
                optax.sgd(learning_rate=self.scheduler),
            )

        dynamic, static = eqx.partition(self.metaOptimizer, trainable_mask)
        self.opt_state = self.optimizer.init(dynamic)
        self.metaOptimizer = eqx.combine(dynamic, static)
        if self.jaxMetaLearnerOptions.load_model is not None and self.jaxMetaLearnerOptions.load_optimizer:
            self.opt_state = eqx.tree_deserialise_leaves(
                self.jaxMetaLearnerOptions.load_model + "/meta_learner_optimizer.eqx", self.opt_state
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
                #+ str(modelOptions.minSlowTau)
                #+ "/"
                #+ str(modelOptions.maxSlowTau)
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            os.makedirs(self.result_directory, exist_ok=False)
            with open(self.result_directory + "/arguments.txt", "w") as f:
                f.writelines(str(self.jaxMetaLearnerOptions))
                f.writelines(str(modelOptions))

            self.average_window = 10
            self.plot = Plot(self.result_directory, self.average_window)

    def chemicals_init(self, rnn, key2):
        layers = rnn.layers
        synaptic_weights = []
        for i in layers:
            # self.synaptic_weights.append(jnp.empty((numberOfChemicals, i.in_features, i.out_features)))
            ## Initialize to xavier initialization
            if self.jaxMetaLearnerOptions.chemicalInitialization == chemicalEnum.same:
                temp = jax.nn.initializers.xavier_uniform()(key2, (i.in_features, i.out_features))
                holder = jnp.tile(temp, (self.numberOfChemicals, 1, 1))
            elif self.jaxMetaLearnerOptions.chemicalInitialization == chemicalEnum.different:
                holder = jax.nn.initializers.xavier_uniform()(
                    key2, (self.numberOfChemicals, i.in_features, i.out_features)
                )
            synaptic_weights.append(holder)
        return tuple(synaptic_weights)

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

        # Dynamically fetch (r.forward1, r.forward2, ...) and pair them with new_parameters
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
            pass
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
            pass
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

    def compute_meta_loss(
        self,
        trainable_metaOptimizer,
        fixed_metaOptimizer,
        synaptic_weights,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
    ):

        metaOptimizer = eqx.combine(trainable_metaOptimizer, fixed_metaOptimizer)
        synaptic_weights_tuple = synaptic_weights
        parameters_tuple = rnn.layers
        rnn = rnn

        # -- inner loop --
        checkpointed_inner_loop = jax.checkpoint(self.full_inner_loop)
        carry_init = (
            synaptic_weights_tuple,
            parameters_tuple,
            rnn,
            metaOptimizer,
        )

        (new_synaptic_weights, new_parameters, new_rnn, _), (y, current_hidden_state) = jax.lax.scan(
            checkpointed_inner_loop,
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

        if self.jaxMetaLearnerOptions.optimizer_mode == JaxOptimizerModeEnum.SOFO:
            return qry_predictions, acc
        return avg_loss, acc

    def get_trainable_mask(self, model):

        mask = jax.tree_util.tree_map(lambda _: False, model)

        mask = eqx.tree_at(lambda m: m.Q_matrix, mask, True)
        mask = eqx.tree_at(lambda m: m.K_matrix, mask, True)
        mask = eqx.tree_at(lambda m: m.v_vector, mask, True)

        return mask

    @eqx.filter_jit
    def make_step(
        self,
        dynamic_model,
        static_model,
        synaptic_weights_tuple,
        rnn,
        x_trn,
        y_trn,
        x_qry,
        y_qry,
        opt_state,
    ):

        if self.jaxMetaLearnerOptions.optimizer_mode == JaxOptimizerModeEnum.BPTT:
            batched_val_and_grad = eqx.filter_vmap(
                jax.value_and_grad(self.compute_meta_loss, has_aux=True),
                in_axes=(
                    None,                   # dynamic_model (Not batched)
                    None,                   # static_model (Not batched)
                    0,                      # synaptic_weights_tuple (Batched)
                    0,                      # rnn (Batched)
                    0,                      # x_trn (Batched)
                    0,                      # y_trn (Batched)
                    0,                      # x_qry (Batched)
                    0,                      # y_qry (Batched)
                )
            )
            (losses, accs), grads = batched_val_and_grad(
                dynamic_model, 
                static_model, 
                synaptic_weights_tuple,
                rnn, 
                x_trn, 
                y_trn, 
                x_qry, 
                y_qry
            )
            grads = jax.tree.map(lambda g: jax.numpy.mean(g, axis=0), grads)
            loss = jnp.mean(losses, axis=0)
            acc = jnp.mean(accs, axis=0)
        elif self.jaxMetaLearnerOptions.optimizer_mode == JaxOptimizerModeEnum.SOFO:
            rng, key = jax.random.split(self.key3)
            v = sample_v(self.jaxMetaLearnerOptions.sofo_samples, dynamic_model, key, identity_sampling=self.jaxMetaLearnerOptions.sofo_identity_sampling)

            def single_task_sofo_geometry(act_params, stat_model, syn_weights, rnn_instance, x_trn_batch, y_trn_batch, x_qry_batch, y_qry_batch):
                def f_active(active_params):
                    d_model = active_params
                    return self.compute_meta_loss(
                        d_model, stat_model, syn_weights, rnn_instance, 
                        x_trn_batch, y_trn_batch, x_qry_batch, y_qry_batch
                    )

                one_hot_targets = jax.nn.one_hot(y_qry_batch.flatten(), num_classes=self.jaxMetaLearnerOptions.output_size)

                def sofo_loss_fn(logits):
                    return optax.safe_softmax_cross_entropy(logits, one_hot_targets).mean()

                outs, tangents_out = jmp(f_active, act_params, v)
                predictions_tangent = tangents_out[0]
                        
                losses, vg = jmp(sofo_loss_fn, outs[0][0], predictions_tangent)

                vggv = jnp.mean(
                        jax.vmap(ggn_ce, in_axes=(1, 0))(predictions_tangent, jax.nn.softmax(outs[0][0], axis=-1)),
                        axis=0
                    )
                acc = outs[1][0]
                return losses[0], acc, vg, vggv
            
            batched_sofo_geometry = eqx.filter_vmap(
                single_task_sofo_geometry,
                in_axes=(
                    None,  # active_params (Not batched)
                    None,  # stat_model (Not batched)
                    0,  # synaptic_weights_tuple (Batched)
                    0,  # rnn (Batched)
                    0,  # x_trn (Batched)
                    0,  # y_trn (Batched)
                    0,  # x_qry (Batched)
                    0,  # y_qry (Batched)
                )
            )
            losses, accs, vgs, vggvs = batched_sofo_geometry(
                dynamic_model,
                static_model,
                synaptic_weights_tuple,
                rnn,
                x_trn,
                y_trn,
                x_qry,
                y_qry
            )

            vg = jax.tree.map(lambda vg: jnp.mean(vg, axis=0), vgs)
            vggv = jnp.mean(vggvs, axis=0)
            loss = jnp.mean(losses, axis=0)
            acc = jnp.mean(accs, axis=0)

            u, s, _ = jnp.linalg.svd(vggv)
            damped_s = s + self.jaxMetaLearnerOptions.sofo_damping * jnp.max(s)
            vggv_vg = (u / damped_s) @ (u.T @ vg)
            grads = jax.tree.map(lambda vs: jnp.dot(jnp.moveaxis(vs,0,-1), vggv_vg), v)
        elif self.jaxMetaLearnerOptions.optimizer_mode == JaxOptimizerModeEnum.RTRL:
            def single_task_rtrl_geometry(act_params, stat_model, syn_weights, rnn_instance, x_trn_batch, y_trn_batch, x_qry_batch, y_qry_batch):
                
                def f_forward(active_params):
                    d_model = active_params
                    avg_loss, acc = self.compute_meta_loss(
                        d_model, stat_model, syn_weights, rnn_instance, 
                        x_trn_batch, y_trn_batch, x_qry_batch, y_qry_batch
                    )
                    return avg_loss, (avg_loss, acc)

                grads, (loss, acc) = jax.jacfwd(f_forward, has_aux=True)(act_params)
                
                return loss, acc, grads

            # 2. Vectorize across tasks using Equinox filter_vmap
            batched_rtrl_geometry = eqx.filter_vmap(
                single_task_rtrl_geometry,
                in_axes=(
                    None,  # active_params (Not batched)
                    None,  # stat_model (Not batched)
                    0,     # synaptic_weights_tuple (Batched)
                    0,     # rnn (Batched)
                    0,     # x_trn (Batched)
                    0,     # y_trn (Batched)
                    0,     # x_qry (Batched)
                    0,     # y_qry (Batched)
                )
            )

            losses, accs, task_grads = batched_rtrl_geometry(
                dynamic_model,
                static_model,
                synaptic_weights_tuple,
                rnn,
                x_trn,
                y_trn,
                x_qry,
                y_qry
            )

            # 3. Average across tasks to get the final update vectors
            loss = jnp.mean(losses, axis=0)
            acc = jnp.mean(accs, axis=0)
            grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), task_grads)

        grads_filtered = eqx.filter(grads, eqx.is_array)

        if self.jaxMetaLearnerOptions.optimizer_mode == JaxOptimizerModeEnum.SOFO:
            grad_norm = jnp.max(s)
        else:
            grad_norm = optax.global_norm(grads_filtered)

        updates, new_opt_state = self.optimizer.update(grads_filtered, opt_state, dynamic_model)
        new_dynamic_model = optax.apply_updates(dynamic_model, updates)

        return new_dynamic_model, new_opt_state, loss, acc, grad_norm

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
                x_trn = jnp.expand_dims(x_trn, 0)
                y_trn = jnp.expand_dims(y_trn, 0)
                x_qry = jnp.expand_dims(x_qry, 0)
                y_qry = jnp.expand_dims(y_qry, 0)
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
            def full_rnns_setup(rnn, key):
                rnn = rnn.reset_weights(key)
                synaptic_weights = self.chemicals_init(rnn, key)
                new_parameters = list(rnn.layers)
                new_synaptic_weights = list(synaptic_weights)
                for idx, parameter in enumerate(rnn.layers):
                    synaptic_weight = synaptic_weights[idx]
                    new_synaptic_weight, new_parameter = self.metaOptimizer.initialize_parameters(
                        synaptic_weight, parameter
                    )
                    new_synaptic_weights[idx] = new_synaptic_weight
                    new_parameters[idx] = new_parameter
                synaptic_weights = tuple(new_synaptic_weights)
                new_parameters = tuple(new_parameters)
                num_layers = len(new_parameters) 
                rnn = eqx.tree_at(
                    lambda r: (r.layers, *(getattr(r, f"forward{i+1}") for i in range(num_layers))),
                    rnn,
                    (new_parameters, *new_parameters),
                )
                return rnn, synaptic_weights

            if self.jaxMetaLearnerOptions.random_architectures:
                key, self.key3 = jax.random.split(self.key3)
                random_int = jax.random.randint(key, (), 0, 2)

                if random_int == 0:
                    self.rnns, self.synaptic_weights = eqx.filter_vmap(full_rnns_setup)(self.rnn_rnns, self.key2_split)
                    self.jaxMetaLearnerOptions.number_of_time_steps = self.true_number_of_timesteps
                else:
                    self.rnns, self.synaptic_weights = eqx.filter_vmap(full_rnns_setup)(self.rnn_ffs, self.key2_split)
                    self.jaxMetaLearnerOptions.number_of_time_steps = 1
            else:
                self.rnns, self.synaptic_weights = eqx.filter_vmap(full_rnns_setup)(self.rnns, self.key2_split) 

            # -- meta-optimization --
            trainable_mask = self.get_trainable_mask(self.metaOptimizer)
            dynamic_model, static_model = eqx.partition(self.metaOptimizer, trainable_mask)
            new_dynamic_model, self.opt_state, loss, acc, grad_norm = self.make_step(
                dynamic_model,
                static_model,
                self.synaptic_weights,
                self.rnns,
                x_trn,
                y_trn,
                x_qry,
                y_qry,
                self.opt_state,
            )

            self.metaOptimizer = eqx.combine(new_dynamic_model, static_model)

            if self.save_results:
                log([loss.item()], self.result_directory + "/loss_meta.txt")
                log([acc], self.result_directory + "/acc_meta.txt")

            line = "Train Episode: {}\tLoss: {:.6f}\tAccuracy: {:.3f}\tGrad Norm: {:.6f}\tTraining/Class: {}".format(
                eps + 1, loss.item(), acc, grad_norm, current_training_data_per_class
            )
            if self.jaxMetaLearnerOptions.random_architectures:
                line += "\tRandom Int: {}".format(random_int)
            if self.jaxMetaLearnerOptions.display:
                print(line)

            if self.save_results:

                with open(self.result_directory + "/params.txt", "a") as f:
                    f.writelines(line + "\n")

                if eps % 10 == 0:
                    self.metaOptimizer.Q_matrix.block_until_ready()
                    with open(self.result_directory + "/{}.txt".format("Q_matrix"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.Q_matrix)))
                    with open(self.result_directory + "/{}.txt".format("K_matrix"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.K_matrix)))
                    with open(self.result_directory + "/{}.txt".format("z_vector"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.z_vector)))
                    with open(self.result_directory + "/{}.txt".format("y_vector"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.y_vector)))
                    with open(self.result_directory + "/{}.txt".format("v_vector"), "a") as f:
                        f.writelines("Episode: {}: {} \n".format(eps + 1, np.array(self.metaOptimizer.v_vector)))
            if jnp.isnan(loss):
                #raise ValueError("Loss is NaN")
                print("Loss is NaN")
                break


        if self.save_results:
            self.plot()
            # -- save model --
            eqx.tree_serialise_leaves(self.result_directory + "/meta_learner_model.eqx", self.metaOptimizer)
            # -- save optimizer state --
            eqx.tree_serialise_leaves(self.result_directory + "/meta_learner_optimizer.eqx", self.opt_state)

        print("Training completed.")
        #self.make_step.clear_cache()


def main_jax_rnn_meta_learner():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # second gpu
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #initialise_tracking()
    #jax.config.update("jax_debug_nans", True)
    key = jax.random.PRNGKey(42)
    outer_location = os.getcwd() + r"/results_4/mode_9_scalar_converted_13_chems_extended_full_sweep"
    tau_mins = os.listdir(outer_location)
    tau_mins = tau_mins[8:]  # skip the first one which is tau_min=1
    for tau_min in tau_mins:
        tau_min_location = os.path.join(outer_location, tau_min)
        tau_maxs = os.listdir(tau_min_location)
        for tau_max in tau_maxs:
            full_location = os.path.join(tau_min_location, tau_max)
    
            
            # jax.config.update("jax_enable_x64", False)

            # -- load data
            numWorkers = 2
            epochs = 5000

            dataset_name = "EMNIST"
            minTrainingDataPerClass = 5
            maxTrainingDataPerClass = 70
            queryDataPerClass = 20
            numberOfTimeSteps = 28
            batch_size = 1

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
                queryDataPerClass = 50
                dataset = AddBernoulliTaskDataset(
                    minSequenceLength=minTrainingDataPerClass,
                    maxSequenceLength=maxTrainingDataPerClass,
                    querySequenceLength=50,
                )
                dimOut = 2
                dimIn = 2
                numberOfClasses = 1
            elif dataset_name == "IMDB":
                numberOfClasses = 2
                dimOut = 2
                queryDataPerClass = 10
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
                dimOut = 4
                queryDataPerClass = 10
                dataset = IMDBWord2VecMetaDataset(
                    minNumberOfSequences=minTrainingDataPerClass,
                    maxNumberOfSequences=maxTrainingDataPerClass,
                    query_q=queryDataPerClass,
                    max_seq_len=200,
                )
                numWorkers = 0
                dimIn = 300

            sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=epochs * numberOfClasses * batch_size)
            metatrain_dataset = DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=numberOfClasses * batch_size,
                drop_last=True,
                num_workers=numWorkers,
                persistent_workers=False,
            )

            # -- options
            modelOptions = None
            modelOptions = fastRnnOptions(
                nonLinear=JaxActivationNonLinearEnum.tanh,
                update_rules=[0, 1, 2, 4, 9, 12],  # 4
                minSlowTau=int(tau_min),
                maxSlowTau=int(tau_max),
                y_vector=yVectorEnum.none,
                z_vector=zVectorEnum.default,
                operator=operatorEnum.mode_9,
            )
            
            # device = "cpu"
            current_dir = os.getcwd()
            continue_training = (
                current_dir + "/results_4/Jax_rnn_fixed_28/20260603-160027" #"/results_4/mode_9_scalar_converted_13_chems"
                #+ "/results_4/jax_rnn_fixed_7/20260525-021101"
                #+ "/results_4/jax_rnn_fixed_7/20260525-021204"
                #current_dir + "/results_4/mode_9_rand_converted"
                #+ "/results_3/jax_rnn_1_chem/20260423-005009"
                #+ "/results_3/jax_rnn_9_chems_100/20260422-175900"
            )  # "/results_2/jax_rnn_7_DSEF_fixed/20260217-174916" # "/results_2/jax_rnn_12/20260121-024411"#"/results_2/jax_rnn_12_28/20260126-043934"
            # -- meta-learner options
            metaLearnerOptions = JaxRnnMetaLearnerOptions(
                seed=42,
                save_results=True,
                results_subdir="Jax_random_architectures",
                metatrain_dataset=dataset_name,
                display=True,
                metaLearningRate=0.001,
                numberOfClasses=numberOfClasses,
                dataset_name=dataset_name,
                chemicalInitialization=chemicalEnum.same,
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
                load_model=continue_training,
                load_optimizer=False,
                dont_load_z_y=False,
                error_type=JaxErrorTypeEnum.DFA,
                low_dim_DFA=-1,
                two_layer_RNN=False,
                feedforward=False,
                meta_optimizer=JaxMetaOptimizerEnum.Adam,
                optimizer_mode=JaxOptimizerModeEnum.BPTT,
                sofo_samples=65,
                sofo_damping=1e-6,
                sofo_identity_sampling=False,
                batch_size=batch_size,
                random_architectures=True,
            )

            metalearning_model = JaxMetaLearnerRNN(
                modelOptions=modelOptions,
                jaxMetaLearnerOptions=metaLearnerOptions,
                key=key,
                numberOfChemicals=5,
                metaTrainingDataset=metatrain_dataset,
            )

            metalearning_model.train()
            exit()
          
