"""Export the active Equinox meta-learner checkpoint as readable text."""

from pathlib import Path

import equinox as eqx
import numpy as np

from options.complex_options import operatorEnum, yVectorEnum, zVectorEnum
from options.fast_rnn_options import fastRnnOptions
from options.jax_rnn_meat_learner_options import JaxActivationNonLinearEnum
from synapses.jax_fast_rnn import JAXFastRnn


def main() -> None:
    checkpoint_dir = (
        Path(__file__).resolve().parents[1]
        / "results_4"
        / "Jax_13_chem_DSEF_full_sweep_200"
        / "1"
        / "400"
    )
    checkpoint = checkpoint_dir / "meta_learner_model.eqx"
    output = checkpoint_dir / "meta_learner_parameters.txt"

    options = fastRnnOptions(
        nonLinear=JaxActivationNonLinearEnum.tanh,
        update_rules=[0, 1, 2, 3, 4, 9, 12],
        minSlowTau=2,
        maxSlowTau=50,
        y_vector=yVectorEnum.none,
        z_vector=zVectorEnum.default,
        operator=operatorEnum.mode_9,
    )
    template = JAXFastRnn(numberOfChemicals=13, fastRnnOptionsVal=options)
    model = eqx.tree_deserialise_leaves(str(checkpoint), template)

    arrays = (
        ("Q_matrix", model.Q_matrix),
        ("K_matrix", model.K_matrix),
        ("v_vector", model.v_vector),
        ("y_vector", model.y_vector),
        ("z_vector", model.z_vector),
    )
    with output.open("w", encoding="utf-8", newline="\n") as report:
        report.write("Meta-trained parameters\n")
        report.write(f"Checkpoint: {checkpoint.relative_to(checkpoint.parents[3])}\n")
        report.write(f"Number of chemicals: {model.numberOfChemicals}\n")
        report.write(f"Update rules: {options.update_rules}\n")
        report.write("\nThese are the learned parameter arrays used by JAXFastRnn.\n")
        for name, value in arrays:
            array = np.asarray(value)
            report.write(f"\n{name} (shape {array.shape})\n")
            report.write(np.array2string(array, precision=10, suppress_small=False, max_line_width=120))
            report.write("\n")

    print(f"Exported meta-trained parameters to {output}")


if __name__ == "__main__":
    main()
