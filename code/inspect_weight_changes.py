import argparse
from pathlib import Path

import numpy as np


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results_4" / "inner_loop" / "70"


def main():
    parser = argparse.ArgumentParser(
        description="Load sampled weight changes and select layer, time, input, and output indexes."
    )
    parser.add_argument(
        "episode",
        nargs="?",
        default="0017",
        help="Episode number, e.g. 0001 (default: 0001).",
    )
    parser.add_argument("image", nargs="?", type=int, default=10, help="Zero-based training-image index (default: 0).")
    parser.add_argument("input_synapse", nargs="?", type=int, default=4, help="Zero-based input-synapse index (default: 0).")
    parser.add_argument("output_neuron", nargs="?", type=int, default=4, help="Zero-based output-neuron index (default: 0).")
    parser.add_argument("--layer", type=int, default=4, help="Zero-based layer index (default: 0).")
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
        parser.error("indexes must be integers")

    path = args.file if args.file is not None else args.directory / f"weight_changes_episode_{args.episode}.npy"
    if not path.is_file():
        parser.error(f"Array file not found: {path}")

    changes = np.load(path)
    if changes.dtype == object:
        layers = list(changes)
    elif changes.ndim == 3:
        # Legacy format: one layer, all training-image weight changes.
        layers = [changes]
    elif changes.ndim == 4:
        # Current format: (layers, sampled images, padded inputs, padded outputs).
        shape_path = path.with_name(path.stem + "_layer_shapes.npy")
        if not shape_path.is_file():
            parser.error(f"Layer shape metadata file not found: {shape_path}")
        layer_shapes = np.load(shape_path)
        layers = [
            changes[layer, :, : int(rows), : int(cols)]
            for layer, (rows, cols) in enumerate(layer_shapes)
        ]
    elif changes.ndim == 5:
        # (tasks, sampled images, layers, inputs, outputs)
        layers = [changes[0, :, layer] for layer in range(changes.shape[2])]
    else:
        parser.error(f"Expected a weight-change array with layer and time axes, got shape {changes.shape}")

    if not 0 <= args.layer < len(layers):
        parser.error(f"Layer index {args.layer} is outside valid range 0..{len(layers) - 1}")
    layer_changes = layers[args.layer]
    if layer_changes.ndim != 3:
        parser.error(f"Expected selected layer shape (sampled images, inputs, outputs), got {layer_changes.shape}")

    index = (image, input_synapse, output_neuron)
    try:
        value = layer_changes[index]
    except IndexError:
        parser.error(f"Index {index} is outside array shape {changes.shape}")

    print(f"File: {path}")
    print(f"Saved shape: {changes.shape}")
    print(f"Layer {args.layer} shape: {layer_changes.shape}")
    print(f"weight_changes[layer={args.layer}, sampled_time={image}, input={input_synapse}, output={output_neuron}] = {value}")


if __name__ == "__main__":
    main()
