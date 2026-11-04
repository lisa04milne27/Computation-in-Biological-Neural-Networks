import argparse
from pathlib import Path

import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results_4" / "inner_loop" / "70"


def main():
    parser = argparse.ArgumentParser(
        description="Load a saved weight-change array and select elements by index."
    )
    parser.add_argument(
        "episode",
        nargs="?",
        default="0001",
        help="Episode number, e.g. 0001 (default: 0001).",
    )
    parser.add_argument("image", nargs="?", type=int, help="Zero-based training-image index.")
    parser.add_argument("input_synapse", nargs="?", type=int, help="Zero-based input-synapse index.")
    parser.add_argument("output_neuron", nargs="?", type=int, help="Zero-based output-neuron index.")
    parser.add_argument("--image", dest="image_option", type=int, help="Training-image index.")
    parser.add_argument("--input-synapse", dest="input_synapse_option", type=int, help="Input-synapse index.")
    parser.add_argument("--output-neuron", dest="output_neuron_option", type=int, help="Output-neuron index.")
    parser.add_argument("--file", type=Path, help="Explicit .npy file path (overrides episode/directory).")
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Folder containing weight_changes_episode_*.npy (default: {DEFAULT_RESULTS}).",
    )
    args = parser.parse_args()

    image = args.image_option if args.image_option is not None else args.image
    input_synapse = args.input_synapse_option if args.input_synapse_option is not None else args.input_synapse
    output_neuron = args.output_neuron_option if args.output_neuron_option is not None else args.output_neuron
    if image is None or input_synapse is None or output_neuron is None:
        parser.error("provide image, input-synapse, and output-neuron indexes")

    path = args.file if args.file is not None else args.directory / f"weight_changes_episode_{args.episode}.npy"
    if not path.is_file():
        parser.error(f"Array file not found: {path}")

    changes = np.load(path)
    if changes.ndim != 3:
        parser.error(f"Expected a 3D array (images, input synapses, output neurons), got {changes.shape}")

    index = (image, input_synapse, output_neuron)
    try:
        value = changes[index]
    except IndexError:
        parser.error(f"Index {index} is outside array shape {changes.shape}")

    print(f"File: {path}")
    print(f"Shape: {changes.shape}")
    print(f"weight_changes{index} = {value}")


if __name__ == "__main__":
    main()
