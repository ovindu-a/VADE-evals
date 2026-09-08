"""Model-id -> ModelAdapter lookup. Add a new model family by writing its
adapter file and registering a match here -- nothing in common/ or das/
needs to change."""
from .qwen2_5_vl import Qwen25VLAdapter


def get_adapter(model_id):
    if "qwen2.5-vl" in model_id.lower() or "qwen2_5_vl" in model_id.lower():
        return Qwen25VLAdapter(model_id)
    raise ValueError(
        f"No ModelAdapter registered for model_id={model_id!r}. "
        f"Add one under methods/adapters/ and register it in {__file__}."
    )
