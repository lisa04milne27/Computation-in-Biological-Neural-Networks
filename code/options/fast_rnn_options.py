from typing import Union

import equinox as eqx
from options.complex_options import (
    nonLinearEnum,
    operatorEnum,
    yVectorEnum,
    zVectorEnum,
)
from options.jax_rnn_meat_learner_options import JaxActivationNonLinearEnum


class fastRnnOptions(eqx.Module):
    """
    Options for the complex synapse and individual complex synapse
    """

    nonLinear: Union[nonLinearEnum, JaxActivationNonLinearEnum] = eqx.field(static=True)
    update_rules: tuple[bool] = eqx.field(static=True)
    minSlowTau: int = eqx.field(static=True)
    maxSlowTau: int = eqx.field(static=True)
    y_vector: yVectorEnum = eqx.field(static=True)
    z_vector: zVectorEnum = eqx.field(static=True)
    operator: operatorEnum = eqx.field(static=True)

    def __init__(
        self,
        nonLinear: Union[nonLinearEnum, JaxActivationNonLinearEnum] = nonLinearEnum.tanh,
        update_rules=None,
        minSlowTau: int = 2,
        maxSlowTau: int = 50,
        y_vector: yVectorEnum = yVectorEnum.none,
        z_vector: zVectorEnum = zVectorEnum.all_ones,
        operator: operatorEnum = operatorEnum.mode_6,
    ):
        self.nonLinear = nonLinear
        self.update_rules = tuple(update_rules) if update_rules is not None else ()
        self.minSlowTau = minSlowTau
        self.maxSlowTau = maxSlowTau
        self.y_vector = y_vector
        self.z_vector = z_vector
        self.operator = operator

    def __str__(self):
        string = ""
        fields = ["nonLinear", "update_rules", "minSlowTau", "maxSlowTau", "y_vector", "z_vector", "operator"]
        for field in fields:
            string += f"{field}: {getattr(self, field)}\n"
        return string