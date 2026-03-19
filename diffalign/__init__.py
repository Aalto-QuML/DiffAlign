def __getattr__(name):
    if name == "DiffAlignModel":
        from diffalign.model import DiffAlignModel
        return DiffAlignModel
    if name == "predict_precursors":
        from diffalign.inference import predict_precursors
        return predict_precursors
    if name == "predict_precursors_from_diffalign":
        from diffalign.inference import predict_precursors_from_diffalign
        return predict_precursors_from_diffalign
    raise AttributeError(f"module 'diffalign' has no attribute {name!r}")
