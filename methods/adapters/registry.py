"""Model-id -> ModelAdapter lookup. Add a new model family by writing its
adapter file and registering a match here -- nothing in common/ or das/
needs to change."""
from .idefics3 import Idefics3Adapter
from .internvl import InternVLAdapter
from .llava import LlavaAdapter
from .qwen2_5_vl import Qwen25VLAdapter


def get_adapter(model_id):
    if "qwen2.5-vl" in model_id.lower() or "qwen2_5_vl" in model_id.lower():
        return Qwen25VLAdapter(model_id)
    if "internvl" in model_id.lower():
        # Only the native transformers port (`-hf`); the trust_remote_code checkpoints have a
        # different module tree and chat format.
        assert model_id.lower().endswith("-hf"), f"{model_id}: use the transformers-native '-hf' checkpoint"
        return InternVLAdapter(model_id)
    if "idefics3" in model_id.lower():
        return Idefics3Adapter(model_id)
    if "llava-1.5" in model_id.lower() or "llava-v1.5" in model_id.lower():
        # LLaVA-1.5 only: 1.6 / NeXT tile images (anyres) and OneVision has a Qwen2 language model.
        assert model_id.lower().endswith("-hf"), f"{model_id}: use the transformers-native llava-hf checkpoint"
        return LlavaAdapter(model_id)
    raise ValueError(
        f"No ModelAdapter registered for model_id={model_id!r}. "
        f"Add one under methods/adapters/ and register it in {__file__}."
    )
