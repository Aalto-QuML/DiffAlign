from rdkit import Chem
import random

MOCK_PRECURSORS = {
    'CC(=O)Oc1ccccc1C(=O)O': [  # Aspirin
        ('O=C(O)c1ccccc1O', 'CC(=O)OC(=O)C'),
        ('c1ccccc1O', 'CC(=O)Cl'),
    ],
    'CC(=O)Nc1ccc(O)cc1': [  # Paracetamol
        ('Nc1ccc(O)cc1', 'CC(=O)O'),
        ('Nc1ccc(O)cc1', 'CC(=O)Cl'),
    ],
}

def predict_precursors(
    product_smiles: str,
    n_precursors: int = 5,
    diffusion_steps: int = 100,
    temperature: float = 1.0,
    beam_size: int = 10,
) -> list[dict]:
    """Mock prediction function - replace with your actual model."""
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return []
    results = []
    if product_smiles in MOCK_PRECURSORS:
        for precursor_set in MOCK_PRECURSORS[product_smiles][:n_precursors]:
            score = random.uniform(0.7, 0.95)
            results.append({'precursors': '.'.join(precursor_set), 'score': score})
    while len(results) < n_precursors:
        fake_precursors = random.sample([
            'CCO', 'CC(=O)O', 'c1ccccc1', 'CC(C)C', 'CCN', 'CCCO',
            'c1ccc(O)cc1', 'CC(=O)Cl', 'CCBr', 'C=CC=C', 'CC#N',
            'c1ccc(N)cc1', 'OC(=O)C=C', 'CC(=O)OC(=O)C', 'ClCCCl'
        ], k=random.randint(2, 3))
        score = random.uniform(0.3, 0.7) * (temperature / 1.0)
        results.append({'precursors': '.'.join(fake_precursors), 'score': min(score, 0.99)})
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:n_precursors]