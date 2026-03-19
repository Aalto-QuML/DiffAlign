def __getattr__(name):
    if name in ("GraphTransformerWithY", "GraphTransformerWithYAtomMapPosEmb"):
        from diffalign.neuralnet.transformer_model_with_y import (
            GraphTransformerWithY,
            GraphTransformerWithYAtomMapPosEmb,
        )
        return {"GraphTransformerWithY": GraphTransformerWithY,
                "GraphTransformerWithYAtomMapPosEmb": GraphTransformerWithYAtomMapPosEmb}[name]
    raise AttributeError(f"module 'diffalign.neuralnet' has no attribute {name!r}")
