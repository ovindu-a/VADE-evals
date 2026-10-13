"""Model-id -> ModelAdapter lookup. Add a new model family by writing its
adapter file and registering a match here -- nothing in common/ or das/
needs to change."""
from .internvl import InternVLAdapter
from .qwen2_5_vl import Qwen25VLAdapter


def get_adapter(model_id):
    if "qwen2.5-vl" in model_id.lower() or "qwen2_5_vl" in model_id.lower():
        return Qwen25VLAdapter(model_id)
    if "internvl" in model_id.lower():
        # Only the native transformers port (`-hf`); the trust_remote_code checkpoints have a
        # different module tree and chat format.
        assert model_id.lower().endswith("-hf"), f"{model_id}: use the transformers-native '-hf' checkpoint"
        return InternVLAdapter(model_id)
    raise ValueError(
        f"No ModelAdapter registered for model_id={model_id!r}. "
        f"Add one under methods/adapters/ and register it in {__file__}."
    )
