from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from options.jax_rnn_meat_learner_options import (
    JaxActivationNonLinearEnum,
    JaxErrorTypeEnum,
)


def beta_softplus(x: jnp.ndarray, beta: float = 10) -> jnp.ndarray:
    return jax.nn.softplus(beta * x) / beta


class JAXFeedforwardNN(eqx.Module):
    # Forward Layers
    forward1: eqx.nn.Linear  # 784 -> 170
    forward2: eqx.nn.Linear  # 170 -> 130
    forward3: eqx.nn.Linear  # 130 -> 100
    forward4: eqx.nn.Linear  # 100 -> 70
    forward5: eqx.nn.Linear  # 70 -> dim_out

    # DFA/DSEF Feedback Layers
    feedback1: eqx.nn.Linear
    feedback2: eqx.nn.Linear
    feedback3: eqx.nn.Linear
    feedback4: eqx.nn.Linear
    feedback5: eqx.nn.Linear

    layers: tuple[eqx.nn.Linear, ...]
    feedback_layers: tuple[eqx.nn.Linear, ...]

    gradient: bool = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = eqx.field(static=True)
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
    ):
        keys = jax.random.split(key, 9)

        # -- Initialize Forward Architecture --
        self.forward1 = eqx.nn.Linear(input_size, 170, key=keys[0], use_bias=False)
        self.forward2 = eqx.nn.Linear(170, 130, key=keys[1], use_bias=False)
        self.forward3 = eqx.nn.Linear(130, 100, key=keys[2], use_bias=False)
        self.forward4 = eqx.nn.Linear(100, 70, key=keys[3], use_bias=False)
        self.forward5 = eqx.nn.Linear(70, dim_out, key=keys[4], use_bias=False)

        self.layers = (self.forward1, self.forward2, self.forward3, self.forward4, self.forward5)

        # -- Configure Error Dimensions --
        if error_type == JaxErrorTypeEnum.DFA:
            error_size = dim_out
        elif error_type == JaxErrorTypeEnum.DSEF:
            error_size = 1
        else:
            raise ValueError(f"Unsupported error type: {error_type}")

        # -- Initialize Feedback Architecture --
        self.feedback1 = eqx.nn.Linear(error_size, 784, key=keys[5], use_bias=False)
        self.feedback2 = eqx.nn.Linear(error_size, 170, key=keys[5], use_bias=False)
        self.feedback3 = eqx.nn.Linear(error_size, 130, key=keys[6], use_bias=False)
        self.feedback4 = eqx.nn.Linear(error_size, 100, key=keys[7], use_bias=False)
        self.feedback5 = eqx.nn.Linear(error_size, 70, key=keys[8], use_bias=False)

        self.feedback_layers = (self.feedback1, self.feedback2, self.feedback3, self.feedback4, self.feedback5)

        # -- State and Activations --
        self.gradient = gradient
        self.activation = activation
        if self.activation is JaxActivationNonLinearEnum.softplus:
            self.activation = beta_softplus

        self.beta = 10.0  # Default beta for softplus; can be made configurable if needed

        self.error_type = error_type
        self.low_dim_DFA = low_dim_DFA

    def low_dim_feedback_initialization(
        self, key: jax.random.PRNGKey, in_features: int, out_features: int
    ) -> jnp.ndarray:
        a = (18.0 / (self.low_dim_DFA * (in_features + out_features))) ** 0.25
        vec_pre1 = jax.random.uniform(key, (self.low_dim_DFA, in_features), minval=-a, maxval=a)
        vec_pre2 = jax.random.uniform(key, (self.low_dim_DFA, out_features), minval=-a, maxval=a)
        full_pre = jnp.dot(vec_pre1.T, vec_pre2)
        return full_pre.T

    def reset_weights(self, key: jax.random.PRNGKey) -> "JAXFeedforwardNN":
        keys = jax.random.split(key, 9)
        new_self = self

        # 1. Reset Forward Layers
        new_self = eqx.tree_at(lambda r: r.forward1, new_self, eqx.nn.Linear(self.forward1.in_features, self.forward1.out_features, key=keys[0], use_bias=False))
        new_self = eqx.tree_at(lambda r: r.forward2, new_self, eqx.nn.Linear(self.forward2.in_features, self.forward2.out_features, key=keys[1], use_bias=False))
        new_self = eqx.tree_at(lambda r: r.forward3, new_self, eqx.nn.Linear(self.forward3.in_features, self.forward3.out_features, key=keys[2], use_bias=False))
        new_self = eqx.tree_at(lambda r: r.forward4, new_self, eqx.nn.Linear(self.forward4.in_features, self.forward4.out_features, key=keys[3], use_bias=False))
        new_self = eqx.tree_at(lambda r: r.forward5, new_self, eqx.nn.Linear(self.forward5.in_features, self.forward5.out_features, key=keys[4], use_bias=False))

        # 2. Reset Feedback Layers
        def _reset_fb(layer: eqx.nn.Linear, k: jax.random.PRNGKey) -> eqx.nn.Linear:
            new_fb = eqx.nn.Linear(layer.in_features, layer.out_features, key=k, use_bias=False)
            if self.low_dim_DFA > 0 and self.error_type == JaxErrorTypeEnum.DFA:
                new_fb = eqx.tree_at(
                    lambda r: r.weight,
                    new_fb,
                    self.low_dim_feedback_initialization(k, layer.in_features, layer.out_features)
                )
            elif self.error_type == JaxErrorTypeEnum.DSEF:
                ## Xavier Uniform initialization for DSEF feedbacks
                a = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
                new_fb = eqx.tree_at(
                    lambda r: r.weight,
                    new_fb,
                    jax.random.uniform(k, (layer.out_features, layer.in_features), minval=-a, maxval=a)
                )
            return new_fb

        new_self = eqx.tree_at(lambda r: r.feedback1, new_self, _reset_fb(self.feedback1, keys[5]))
        new_self = eqx.tree_at(lambda r: r.feedback2, new_self, _reset_fb(self.feedback2, keys[6]))
        new_self = eqx.tree_at(lambda r: r.feedback3, new_self, _reset_fb(self.feedback3, keys[7]))
        new_self = eqx.tree_at(lambda r: r.feedback4, new_self, _reset_fb(self.feedback4, keys[8]))
        new_self = eqx.tree_at(lambda r: r.feedback5, new_self, _reset_fb(self.feedback5, keys[9]))

        return new_self
    
    def initialise_hidden_state(self, batch_size: int) -> None:
        # FNN does not utilize hidden state, but method is defined for compatibility with RNN training loops
        return None

    def __call__(
        self, x: jnp.ndarray, h: Optional[jnp.ndarray] = None, label: Optional[jnp.ndarray] = None
    ) -> tuple:
        # Note: `h` is ignored but kept in the signature to avoid breaking standard train loop unpacking.

        # -- FNN Forward Pass --
        a1 = self.forward1(x)
        h1 = self.activation(a1)

        a2 = self.forward2(h1)
        h2 = self.activation(a2)

        a3 = self.forward3(h2)
        h3 = self.activation(a3)

        a4 = self.forward4(h3)
        h4 = self.activation(a4)

        y = self.forward5(h4)

        if label is None:
            return y, None, None, None, None
        
        softmax_y = jax.nn.softmax(y)

        # -- Initial Error Logic --
        if self.error_type == JaxErrorTypeEnum.DFA:
            error = softmax_y - label
        elif self.error_type == JaxErrorTypeEnum.DSEF:
            label_index = jnp.argmax(label, axis=-1, keepdims=True)
            y_index = jnp.take_along_axis(softmax_y, label_index, axis=-1)
            scaler_error = (jnp.squeeze(y_index) <= 0.5).astype(jnp.float32)
            error = scaler_error
        else:
            scaler_error = jnp.array(0.0, dtype=jnp.float32)

        # -- Stop Gradients for Feedbacks --
        w_fb1 = jax.lax.stop_gradient(self.feedback1.weight)
        w_fb2 = jax.lax.stop_gradient(self.feedback2.weight)
        w_fb3 = jax.lax.stop_gradient(self.feedback3.weight)
        w_fb4 = jax.lax.stop_gradient(self.feedback4.weight)
        w_fb5 = jax.lax.stop_gradient(self.feedback5.weight)

        # -- Feedback Connections --
        fb_err1 = jnp.dot(error, w_fb1.T)
        fb_err2 = jnp.dot(error, w_fb2.T)
        fb_err3 = jnp.dot(error, w_fb3.T)
        fb_err4 = jnp.dot(error, w_fb4.T)
        fb_err5 = jnp.dot(error, w_fb5.T)

        # Reset DSEF error to match final standard mapping requirement
        if self.error_type == JaxErrorTypeEnum.DSEF:
            error = softmax_y - label

        # -- Compile Returns --
        if self.gradient:
            # Calculate scalar derivative element-wise
            grad1 = -jnp.expm1(-self.beta * x)
            grad2 = -jnp.expm1(-self.beta * h1)
            grad3 = -jnp.expm1(-self.beta * h2)
            grad4 = -jnp.expm1(-self.beta * h3)
            grad5 = -jnp.expm1(-self.beta * h4)

            # Modulate DFA direct error with local activation gradient
            err1 = fb_err1 * grad1
            err2 = fb_err2 * grad2
            err3 = fb_err3 * grad3
            err4 = fb_err4 * grad4
            err5 = fb_err5 * grad5
            err6 = error  # The last layer utilizes strictly standard error

        else:
            err1, err2, err3, err4, err5, err6 = fb_err1, fb_err2, fb_err3, fb_err4, fb_err5, error

        activations_arr = [
            (x, h1),
            (h1, h2),
            (h2, h3),
            (h3, h4),
            (h4, softmax_y),
        ]

        errors_arr = [(err1, err2), (err2, err3), (err3, err4), (err4, err5), (err5, err6)]

        # Returns `y`, `h_new` (None for FNN), `activations`, `errors`
        return y, None, activations_arr, errors_arr, scaler_error, jnp.squeeze(y_index) if self.error_type == JaxErrorTypeEnum.DSEF else jnp.array(0.0)
