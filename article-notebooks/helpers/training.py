import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import List, Dict

from nnbma.dataset import MaskDataset, RegressionDataset
from nnbma.learning import LearningParameters, learning_procedure
from nnbma.networks import NeuralNetwork

from helpers.preprocessing import prepare_data, build_data_transformers


def procedure(
    lines: List[str],
    model: NeuralNetwork,
    learning_params: LearningParameters,
    mask: bool,
    verbose: bool=True,
) -> Dict[str, object]:
    """TODO"""

    ## Dataset setup

    (
        dataset_train,
        dataset_val,
        dataset_mask_train,
        dataset_mask_val,
    ) = prepare_data(lines)

    if not mask:
        dataset_mask_train = None
        dataset_mask_val = None

    (
        operator_x,
        inverse_operator_x,
        operator_y,
        inverse_operator_y,
    ) = build_data_transformers(dataset_train)

    ## Architecture

    model.inputs_transformer = operator_x
    model.outputs_transformer = inverse_operator_y

    # Test of restriction
    model.eval()
    model.restrict_to_output_subset(dataset_train.outputs_names)
    model.train()

    # Test of copy
    model.copy()

    print(
        f"Number of parameters: {model.count_parameters(learnable_only=False):,} ({model.count_bytes(learnable_only=False, display = True)})"
    )
    print(
        f"Number of learnable parameters: {model.count_parameters():,} ({model.count_bytes(display = True)})"
    )

    ## Normalization

    dataset_train_transf = dataset_train.apply_transf(operator_x, operator_y)
    dataset_val_transf = dataset_val.apply_transf(operator_x, operator_y)

    meth = getattr(model, "update_standardization", None)
    if callable(meth):
        meth(dataset_train_transf.getall()[0])

    ## Learning procedure

    results = learning_procedure(
        model,
        (dataset_train_transf, dataset_val_transf),
        learning_params,
        (dataset_mask_train, dataset_mask_val),
        verbose=verbose,
    )

    return results