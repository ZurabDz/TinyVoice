"""Single-host data-parallel helpers (1D mesh over ('data',))."""

import jax
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def build_mesh() -> Mesh:
    """1D mesh over all visible devices. Trivial no-op on 1-device hosts."""
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    return Mesh(devices, ("data",))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def data_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P("data"))


def shard_trainer_state(trainer: nnx.Module, mesh: Mesh) -> None:
    """Replicate the full nnx state of `trainer` across mesh devices. Idempotent."""
    state = nnx.state(trainer)
    state = jax.device_put(state, replicated_sharding(mesh))
    nnx.update(trainer, state)


def shard_batch(batch, mesh: Mesh):
    """Shard a pytree of host arrays along their leading (batch) axis."""
    return jax.device_put(batch, data_sharding(mesh))
