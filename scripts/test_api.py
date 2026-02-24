import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from api import predict


if __name__ == "__main__":
    predict.predict_precursors_from_diffalign(
        product_smiles="C1=CC=C(C=C1)Cl",
    )