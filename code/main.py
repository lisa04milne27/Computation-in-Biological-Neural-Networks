import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
    print("Multiprocessing start method set to 'spawn'")
except RuntimeError:
    pass  # Method already set

from metalearners.jax_rnn_meta_learner_main import (  # noqa: F401
    main_jax_rnn_meta_learner,
)
from metalearners.jax_rnn_runner import main_jax_runner  # noqa: F401
from metalearners.meta_learner_main import main  # noqa: F401
from metalearners.rnn_meta_learner_main import main_rnn  # noqa: F401
from metalearners.runner import runner_main  # noqa: F401
from metalearners.runner_rnn import main_runner_rnn  # noqa: F401
from misc.chemical_upscale import chemical_upscale  # noqa: F401
from misc.expand_Q import expand_Q_matrix  # noqa: F401
from misc.load_from_params import load_model  # noqa: F401
from misc.torch_jax_converter import torch_to_jax_chemical_nn  # noqa: F401
from nn.backprop import backprop_main  # noqa: F401
from nn.rflo import rflo_main  # noqa: F401
from nn.rflo_2 import rflo_main_2  # noqa: F401
from nn.rnn_backprop import rnn_backprop_main  # noqa: F401

# torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    #main()
    # runner_main()
    # load_model()
    # backprop_main()
    # rnn_backprop_main()
    # main_rnn()
    # main_runner_rnn()
    # rflo_main()
    # rflo_main_2()
    # main_jax_rnn_meta_learner()
    main_jax_runner()
    # chemical_upscale()
    # torch_to_jax_chemical_nn()
    # expand_Q_matrix()
