import jax
import jax.numpy as jnp
from conformer.model import ConformerModel
from flax import nnx
import optax

def compute_mask_v(frames):
    t_mel = (frames - 400) // 160 + 1
    t_conv1 = (t_mel - 3) // 2 + 1
    t_final = (t_conv1 - 3) // 2 + 1
    
    max_frames = 235008
    max_t_mel = (max_frames - 400) // 160 + 1
    max_t_conv1 = (max_t_mel - 3) // 2 + 1
    max_t_final = (max_t_conv1 - 3) // 2 + 1

    real_times = t_final
    mask = jnp.arange(max_t_final) < real_times[:, None]
    mask = jnp.expand_dims(mask, axis=1).repeat(max_t_final, axis=1)
    mask = jnp.expand_dims(mask, axis=1).repeat(4, axis=1)

    return mask, real_times

def test_verification():
    model = ConformerModel(token_count=30, rngs=nnx.Rngs(0))
    batch_size = 2
    max_frames = 235008
    frames = jnp.array([235008, 160000])
    audios = jnp.zeros((batch_size, max_frames))
    
    mask, real_times = compute_mask_v(frames)
    print(f"Mask shape: {mask.shape}")
    print(f"Real times: {real_times}")
    
    out = model(audios, mask=mask, training=False)
    print(f"Output shape: {out.shape}")
    
    assert out.shape[1] == mask.shape[-1], f"Output time dim {out.shape[1]} != mask time dim {mask.shape[-1]}"
    print("Verification successful: Mask and output shapes match!")

if __name__ == "__main__":
    test_verification()
