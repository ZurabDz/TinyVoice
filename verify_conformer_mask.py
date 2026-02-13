import jax
import jax.numpy as jnp
from flax import nnx
import sys
import os

# Ensure we can import conformer
sys.path.append(os.getcwd())

from conformer.model import ConformerEncoder

def test_masking():
    print("Testing ConformerEncoder masking...")
    rngs = nnx.Rngs(0)
    # n_fft=512, hop=160. 16000 samples -> 100 frames.
    # Subsampling: ((100-3)//2+1) -> 49 -> ((49-3)//2+1) -> 24.
    model = ConformerEncoder(token_count=100, d_model=144, num_layers=1, rngs=rngs)
    
    # Batch size 2
    # Item 1: 16000 samples (full)
    # Item 2: 8000 samples (half)
    x = jnp.zeros((2, 16000)) 
    lengths = jnp.array([16000, 8000])

    # Run forward pass
    try:
        # We need to jit specific parts or run eagerly. nnx modules are eager by default unless jitted.
        out, out_lens = model(x, inputs_lengths=lengths, training=False)
        print("Forward pass successful!")
        print(f"Output shape: {out.shape}")
        print(f"Output lengths: {out_lens}")
    except Exception as e:
        print("Forward pass failed:", e)
        import traceback
        traceback.print_exc()
        raise e

    # Check if masking worked implies no NaN (though without mask it might not NaN, just process padding)
    # We can check mask generation logic directly
    
    # Based on our calculation:
    # 16000 -> 24 frames
    # 8000 -> 11 frames (approx, let's trace: 8000/160 = 50. (50-3)//2+1 = 24. (24-3)//2+1 = 11.)
    
    # Let's verify compute_mask
    mask = model.compute_mask(out_lens, out.shape[1])
    print(f"Computed mask shape: {mask.shape}")
    
    # Check Item 1: should be all True (up to 24)
    # If out.shape[1] is 24, then all True. If it padded to 24, it's 24.
    # Actually batches are padded to max length.
    
    # Check Item 2: should be True up to 11, then False
    item2_mask = mask[1, 0, 0, :]
    print(f"Item 2 mask (first 15): {item2_mask[:15]}")
    
    assert jnp.all(item2_mask[:out_lens[1]]), "Mask should be True for valid frames"
    if out.shape[1] > out_lens[1]:
         assert not jnp.any(item2_mask[out_lens[1]:]), "Mask should be False for padding frames"
         print("Mask correctly masks padding!")
    else:
        print("Output length equals max length, so no padding to mask in this batch (or logic differs).")

    print("Verification complete!")

if __name__ == "__main__":
    test_masking()
