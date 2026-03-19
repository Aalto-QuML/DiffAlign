def __getattr__(name):
    if name == "DiscreteDenoisingDiffusionRxn":
        from diffalign.diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn
        return DiscreteDenoisingDiffusionRxn
    raise AttributeError(f"module 'diffalign.diffusion' has no attribute {name!r}")
