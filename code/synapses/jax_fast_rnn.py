import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum


class JAXFastRnn(eqx.Module):
    numberOfChemicals: int = eqx.field(static=True)
    numberUpdateRules: int = eqx.field(static=True)
    update_rules: tuple[bool] = eqx.field(static=True)
    Q_matrix: jnp.ndarray
    K_matrix: jnp.ndarray
    z_vector: jnp.ndarray
    y_vector: jnp.ndarray
    v_vector: jnp.ndarray
    fastRnnOptions: object = eqx.field(static=True)

    def __init__(self, numberOfChemicals: int, fastRnnOptionsVal: object):
        super().__init__()

        self.numberOfChemicals = numberOfChemicals
        self.fastRnnOptions = fastRnnOptionsVal
        self.numberUpdateRules = 14
        self.update_rules = [False] * self.numberUpdateRules

        trueNumberUpdateRules = 0
        for i in self.fastRnnOptions.update_rules:
            trueNumberUpdateRules += 1
            self.update_rules[i] = True
        self.update_rules = tuple(self.update_rules)
        self.numberUpdateRules = trueNumberUpdateRules

        # -- initialize matrices --
        self.Q_matrix = np.zeros((self.numberOfChemicals, self.numberUpdateRules))
        # self.Q_matrix[:, 0] = 1e-3  # rule 0
        self.Q_matrix[:, 0] = 1e-3
        self.Q_matrix = jnp.array(self.Q_matrix)
        self.K_matrix = jnp.zeros((self.numberOfChemicals, self.numberOfChemicals))
        self.z_vector = jnp.zeros((self.numberOfChemicals,))
        self.y_vector = jnp.zeros((self.numberOfChemicals,))
        self.v_vector = jnp.zeros((self.numberOfChemicals,))

        min_tau = self.fastRnnOptions.minSlowTau
        max_tau = self.fastRnnOptions.maxSlowTau

        # -- initialize tau vector --
        base = max_tau / min_tau
        tau_vector = min_tau * jnp.power(base, jnp.linspace(0, 1, self.numberOfChemicals))
        if self.numberOfChemicals == 1:
            self.z_vector = jnp.array([1.0])
            self.y_vector = jnp.array([1.0])
        else:
            self.z_vector = 1.0 / tau_vector
            self.y_vector = 1.0 - self.z_vector

        if self.fastRnnOptions.z_vector == zVectorEnum.all_ones:
            self.z_vector = jnp.ones(self.numberOfChemicals)
        elif self.fastRnnOptions.z_vector == zVectorEnum.default:
            pass


        if self.fastRnnOptions.y_vector == yVectorEnum.last_one:
            self.y_vector[-1] = 1
        elif self.fastRnnOptions.y_vector == yVectorEnum.none:
            pass
        elif self.fastRnnOptions.y_vector == yVectorEnum.first_one:
            self.y_vector[0] = 1
        elif self.fastRnnOptions.y_vector == yVectorEnum.last_one_and_small_first:
            self.y_vector[-1] = 1
            self.y_vector[0] = self.z_vector[-1]
        elif self.fastRnnOptions.y_vector == yVectorEnum.all_ones:
            self.y_vector = jnp.ones(self.numberOfChemicals)
        elif self.fastRnnOptions.y_vector == yVectorEnum.half:
            self.y_vector[-1] = 0.5

    def __call__(
        self,
        synaptic_weight: jnp.ndarray,
        parameter: jnp.ndarray,
        activations_tuple: tuple[jnp.ndarray, jnp.ndarray],
        errors_tuple: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        pre_synaptic_activation = activations_tuple[0]
        post_synaptic_activation = activations_tuple[1]
        pre_synaptic_error = errors_tuple[0]
        post_synaptic_error = errors_tuple[1]
        update_vector = self.calculate_update_vector(
            pre_synaptic_activation,
            post_synaptic_activation,
            pre_synaptic_error,
            post_synaptic_error,
            parameter,
        )
        new_synaptic_weight = jnp.einsum("i,ijk->ijk", self.y_vector, synaptic_weight) + jnp.einsum(
            "i,ijk->ijk",
            self.z_vector,
            self.fastRnnOptions.nonLinear(
                jnp.einsum(
                    "ci,ijk->cjk", self.Q_matrix, update_vector
                )  # This is correct (Q_matrix shape is kinda transposed but left to match previous code)
                + jnp.einsum("ic,ijk->cjk", self.K_matrix, synaptic_weight)
            ),
        )

        if self.fastRnnOptions.operator == operatorEnum.mode_7:
            new_synaptic_norms = jnp.linalg.norm(
                new_synaptic_weight, axis=1, ord=2
            )
            new_synaptic_weight = new_synaptic_weight / (new_synaptic_norms[:, None, :] + 1e-8)

        v_vector_softmax = jax.nn.softmax(self.v_vector)
        new_parameter_weight = jnp.einsum("c,cjk->jk", v_vector_softmax, new_synaptic_weight)

        if self.fastRnnOptions.operator == operatorEnum.mode_7:
            new_parameter_norms = jnp.linalg.norm(new_parameter_weight, axis=0, ord=2)  # TODO check axis
            new_parameter_weight = new_parameter_weight / (new_parameter_norms + 1e-8)
        elif self.fastRnnOptions.operator == operatorEnum.mode_9:
            new_parameter_norms = jnp.linalg.norm(new_parameter_weight, axis=1, ord=2)  # TODO check axis
            new_parameter_weight = new_parameter_weight / (new_parameter_norms[:, None] + 1e-12)
            new_synaptic_weight = new_synaptic_weight / (new_parameter_norms[None, :, None] + 1e-12)

        new_layer = eqx.tree_at(lambda p: p.weight, parameter, new_parameter_weight.T)
        return new_layer, new_synaptic_weight

    def initialize_parameters(
        self, synaptic_weight: jnp.ndarray, parameter: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        synaptic_norm = jnp.linalg.norm(synaptic_weight, axis=1, ord=2)
        new_synaptic_weight = synaptic_weight / synaptic_norm[:, None, :]
        v_vector_softmax = jax.nn.softmax(self.v_vector)
        new_parameter_weight = jnp.einsum("c,cjk->jk", v_vector_softmax, synaptic_weight)
        parameter_norm = jnp.linalg.norm(new_parameter_weight, axis=0, ord=2)
        new_parameter_weight = new_parameter_weight / (parameter_norm + 1e-12)
        new_layer = eqx.tree_at(lambda p: p.weight, parameter, new_parameter_weight.T)
        return new_synaptic_weight, new_layer

    def calculate_update_vector(
        self,
        pre_synaptic_activation: jnp.ndarray,
        post_synaptic_activation: jnp.ndarray,
        pre_synaptic_error: jnp.ndarray,
        post_synaptic_error: jnp.ndarray,
        parameter: jnp.ndarray,
    ) -> jnp.ndarray:

        update_vector = jnp.zeros((self.numberUpdateRules, parameter.in_features, parameter.out_features))
        i = 0

        if self.update_rules[0]:
            update_vector = update_vector.at[i].set(-jnp.outer(post_synaptic_error, pre_synaptic_activation).T)
            i += 1
        if self.update_rules[1]:
            update_vector = update_vector.at[i].set(-jnp.outer(post_synaptic_activation, pre_synaptic_error).T)
            i += 1
        if self.update_rules[2]:
            update_vector = update_vector.at[i].set(-jnp.outer(post_synaptic_error, pre_synaptic_error).T)
            i += 1
        if self.update_rules[3]:
            update_vector = update_vector.at[i].set(-parameter.weight.T)
            i += 1
        if self.update_rules[4]:
            update_vector = update_vector.at[i].set(-jnp.outer(jnp.ones(parameter.out_features), pre_synaptic_error).T)
            i += 1
        if self.update_rules[5]:
            pass
            i += 1
        if self.update_rules[6]:
            pass
            i += 1
        if self.update_rules[7]:
            pass
            i += 1
        if self.update_rules[8]:
            pass
            i += 1
        if self.update_rules[9]:
            update_vector = update_vector.at[i].set(
                (
                    jnp.outer(post_synaptic_activation, pre_synaptic_activation)
                    - jnp.matmul(jnp.outer(post_synaptic_activation, post_synaptic_activation), parameter.weight)
                ).T
            )
            i += 1
        if self.update_rules[10]:
            update_vector = update_vector.at[i].set(-jnp.outer(post_synaptic_error, jnp.ones(parameter.in_features)).T)
            i += 1
        if self.update_rules[11]:
            update_vector = update_vector.at[i].set(
                -jnp.outer(post_synaptic_activation, jnp.ones(parameter.in_features)).T
            )
            i += 1
        if self.update_rules[12]:
            update_vector = update_vector.at[i].set(
                -jnp.outer(jnp.ones(parameter.out_features), pre_synaptic_activation).T
            )
            i += 1
        if self.update_rules[13]:
            pass
            i += 1

        return update_vector
