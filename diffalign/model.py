"""Inference wrapper for the LocalRetro model.

Paper: https://pubs.acs.org/doi/10.1021/jacsau.1c00246
Code: https://github.com/kaist-amsg/LocalRetro

The original LocalRetro code is released under the Apache 2.0 license.
Parts of this file are based on code from the GitHub repository above.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Any, List, Sequence
from typing import Optional
from syntheseus.interface.models import InputType, ReactionType

from syntheseus.interface.molecule import Molecule
from syntheseus.interface.reaction import SingleProductReaction
from syntheseus.reaction_prediction.inference.base import ExternalBackwardReactionModel
from syntheseus.reaction_prediction.utils.inference import (
    get_unique_file_in_dir,
    process_raw_smiles_outputs_backwards,
)
from syntheseus.reaction_prediction.utils.misc import suppress_outputs


from local_retro.scripts.utils import mkdir_p
from local_retro.LocalTemplate.template_decoder import *

class DiffAlignModel(ExternalBackwardReactionModel):
    def __init__(self, *args, **kwargs) -> None:
        """Initializes the LocalRetro model wrapper.

        Assumed format of the model directory:
        - `model_dir` contains the model checkpoint as the only `*.pth` file
        - `model_dir` contains the config as the only `*.json` file
        - `model_dir/data` contains `*.csv` data files needed by LocalRetro
        """
        super().__init__(*args, **kwargs)

    def __call__(
        self, inputs: list[InputType], num_results: Optional[int] = None, reaction_types: int = None
    ) -> list[Sequence[ReactionType]]:
        pass

    def _get_reactions(
        self, inputs: List[Molecule], num_results: int, reaction_types: int = None
    ) -> List[Sequence[SingleProductReaction]]:
        pass