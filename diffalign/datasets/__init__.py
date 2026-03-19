def __getattr__(name):
    if name in ("Dataset", "DataModule", "DatasetInfos"):
        from diffalign.datasets.supernode_dataset import Dataset, DataModule, DatasetInfos
        return {"Dataset": Dataset, "DataModule": DataModule, "DatasetInfos": DatasetInfos}[name]
    raise AttributeError(f"module 'diffalign.datasets' has no attribute {name!r}")
