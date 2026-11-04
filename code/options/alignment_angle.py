import jax
import jax.numpy as jnp

from metalearners.jax_rnn_runner import JaxMetaLearnerRNN


#This will return delta_w_s
def delta_ws():
    delta_ws_list=[]
    return delta_ws_list


def compute_alignment_angle(delta_w_cs, delta_w_bp):
    """
    Computes the angle in degrees between complex synapse weight update 
    and standard backpropagation weight update.
    """
    # 1. Flatten layer weight tensors into 1D vectors
    u = delta_w_cs.reshape(-1)
    v = delta_w_bp.reshape(-1)
    
    # 2. Compute dot product and L2 norms
    dot_product = jnp.dot(u, v)
    norm_u = jnp.linalg.norm(u)
    norm_v = jnp.linalg.norm(v)
    
    # 3. Compute cosine similarity with clipping to avoid numerical range errors [-1, 1]
    cos_sim = jnp.clip(dot_product / (norm_u * norm_v + 1e-8), -1.0, 1.0)

    #I could convert into an angle if I wanted to
    
    return cos_sim