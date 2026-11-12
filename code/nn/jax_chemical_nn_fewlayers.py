from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxErrorTypeEnum,
)


def beta_softplus(x: jnp.ndarray, beta: float = 10.0) -> jnp.ndarray:
    return jax.nn.softplus(beta * x) / beta


class JAXFeedforwardNN(eqx.Module):
    # Forward Layers
    forward1: eqx.nn.Linear  # input_size -> 70
    forward2: eqx.nn.Linear  # 70 -> dim_out

    # DFA/DSEF Feedback Layers
    feedback1: eqx.nn.Linear  # error_size -> input_size
    feedback2: eqx.nn.Linear  # error_size -> 70

    layers: tuple[eqx.nn.Linear, ...]
    feedback_layers: tuple[eqx.nn.Linear, ...]

    gradient: bool = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    activation: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(static=True)
    error_type: JaxErrorTypeEnum = eqx.field(static=True)
    low_dim_DFA: int = eqx.field(static=True)

    def __init__(
        self,
        dim_out: int,
        key: jax.random.PRNGKey,
        input_size: int = 784,
        gradient: bool = False,
        activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        error_type: JaxErrorTypeEnum = JaxErrorTypeEnum.DFA,
        low_dim_DFA: int = -1,
        beta: float = 10.0,
    ):
        keys = jax.random.split(key, 4)

        # -- State and Config --
        self.gradient = gradient
        self.beta = beta
        self.error_type = error_type
        self.low_dim_DFA = low_dim_DFA

        if activation is None or activation is JaxActivationNonLinearEnum.softplus:
            self.activation = lambda x: beta_softplus(x, beta=self.beta)
        else:
            self.activation = activation

        # -- Initialize Forward Architecture --
        self.forward1 = eqx.nn.Linear(input_size, 70, key=keys[0], use_bias=False)
        self.forward2 = eqx.nn.Linear(70, dim_out, key=keys[1], use_bias=False)
        self.layers = (self.forward1, self.forward2)

        # -- Configure Error Dimensions --
        if error_type == JaxErrorTypeEnum.DFA:
            error_size = dim_out
        elif error_type == JaxErrorTypeEnum.DSEF:
            error_size = 1
        else:
            raise ValueError(f"Unsupported error type: {error_type}")

        # -- Initialize Feedback Architecture --
        fb1 = eqx.nn.Linear(error_size, input_size, key=keys[2], use_bias=False)
        fb2 = eqx.nn.Linear(error_size, 70, key=keys[3], use_bias=False)

        self.feedback1 = self._init_feedback_weights(fb1, keys[2])
        self.feedback2 = self._init_feedback_weights(fb2, keys[3])
        self.feedback_layers = (self.feedback1, self.feedback2)

    def _init_feedback_weights(
        self, layer: eqx.nn.Linear, k: jax.random.PRNGKey
    ) -> eqx.nn.Linear:
        if self.low_dim_DFA > 0 and self.error_type == JaxErrorTypeEnum.DFA:
            w = self.low_dim_feedback_initialization(
                k, layer.in_features, layer.out_features
            )
            return eqx.tree_at(lambda r: r.weight, layer, w)
        elif self.error_type == JaxErrorTypeEnum.DSEF:
            a = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
            w = jax.random.uniform(
                k, (layer.out_features, layer.in_features), minval=-a, maxval=a
            )
            return eqx.tree_at(lambda r: r.weight, layer, w)
        return layer

    def low_dim_feedback_initialization(
        self, key: jax.random.PRNGKey, in_features: int, out_features: int
    ) -> jnp.ndarray:
        k1, k2 = jax.random.split(key)
        a = (18.0 / (self.low_dim_DFA * (in_features + out_features))) ** 0.25
        vec_pre1 = jax.random.uniform(
            k1, (self.low_dim_DFA, in_features), minval=-a, maxval=a
        )
        vec_pre2 = jax.random.uniform(
            k2, (self.low_dim_DFA, out_features), minval=-a, maxval=a
        )
        full_pre = jnp.dot(vec_pre1.T, vec_pre2)
        return full_pre.T

    def reset_weights(self, key: jax.random.PRNGKey) -> "JAXFeedforwardNN":
        keys = jax.random.split(key, 4)

        new_f1 = eqx.nn.Linear(
            self.forward1.in_features,
            self.forward1.out_features,
            key=keys[0],
            use_bias=False,
        )
        new_f2 = eqx.nn.Linear(
            self.forward2.in_features,
            self.forward2.out_features,
            key=keys[1],
            use_bias=False,
        )

        new_fb1 = self._init_feedback_weights(
            eqx.nn.Linear(
                self.feedback1.in_features,
                self.feedback1.out_features,
                key=keys[2],
                use_bias=False,
            ),
            keys[2],
        )
        new_fb2 = self._init_feedback_weights(
            eqx.nn.Linear(
                self.feedback2.in_features,
                self.feedback2.out_features,
                key=keys[3],
                use_bias=False,
            ),
            keys[3],
        )

        # Update attributes AND their container tuples simultaneously
        return eqx.tree_at(
            lambda r: (
                r.forward1,
                r.forward2,
                r.layers,
                r.feedback1,
                r.feedback2,
                r.feedback_layers,
            ),
            self,
            (
                new_f1,
                new_f2,
                (new_f1, new_f2),
                new_fb1,
                new_fb2,
                (new_fb1, new_fb2),
            ),
        )

    def initialise_hidden_state(self, batch_size: int) -> None:
        return None

    def __call__(
        self,
        x: jnp.ndarray,
        h: Optional[jnp.ndarray] = None,
        label: Optional[jnp.ndarray] = None,
    ) -> tuple:
        # -- FNN Forward Pass --
        a1 = self.forward1(x)
        h1 = self.activation(a1)
        y = self.forward2(h1)

        if label is None:
            return y, None

        softmax_y = jax.nn.softmax(y)

        # -- Initial Error Logic --
        if self.error_type == JaxErrorTypeEnum.DFA:
            error = softmax_y - label
            scaler_error = jnp.array(0.0, dtype=jnp.float32)
            y_index_out = jnp.array(0.0, dtype=jnp.float32)
        elif self.error_type == JaxErrorTypeEnum.DSEF:
            label_index = jnp.argmax(label, axis=-1, keepdims=True)
            y_index = jnp.take_along_axis(softmax_y, label_index, axis=-1)
            scaler_error = (jnp.squeeze(y_index) <= 0.5).astype(jnp.float32)
            # Keep 1D shape (1,) so dot product with (1, out_features) works cleanly
            error = jnp.atleast_1d(scaler_error)
            y_index_out = jnp.squeeze(y_index)
        else:
            raise ValueError(f"Unsupported error type: {self.error_type}")

        # -- Stop Gradients for Feedbacks --
        w_fb1 = jax.lax.stop_gradient(self.feedback1.weight)  # (784, error_size)
        w_fb2 = jax.lax.stop_gradient(self.feedback2.weight)  # (70, error_size)

        # -- Feedback Connections --
        fb_err1 = jnp.dot(error, w_fb1.T)
        fb_err2 = jnp.dot(error, w_fb2.T)

        # Reset DSEF error to match final standard mapping requirement
        if self.error_type == JaxErrorTypeEnum.DSEF:
            error = softmax_y - label

        # -- Compile Returns for 2-Layer Network --
        if self.gradient:
            # Derivative of beta_softplus(z) is sigmoid(beta * z), which equals 1 - exp(-beta * softplus_out)
            grad1 = -jnp.expm1(-self.beta * x)
            grad2 = -jnp.expm1(-self.beta * h1)

            err1 = fb_err1 * grad1
            err2 = fb_err2 * grad2
            err3 = error
        else:
            err1, err2, err3 = fb_err1, fb_err2, error

        activations_arr = [
            (x, h1),
            (h1, softmax_y),
        ]

        errors_arr = [
            (err1, err2),
            (err2, err3),
        ]

        return y, None, activations_arr, errors_arr, scaler_error, y_index_out