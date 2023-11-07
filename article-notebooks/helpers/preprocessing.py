import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "meudon-pdr", "data")
)

import json
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from helpers.lines import filter_molecules

from nnbma.dataset import MaskDataset, RegressionDataset
from nnbma.operators import (
    ColumnwiseOperator,
    InverseNormalizer,
    Normalizer,
    NormTypes,
    Operator,
    SequentialOperator,
    id,
    log10,
    pow10,
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "meudon-pdr",
    "data",
    "P17r2101E20_all_stat_files_{}.pkl",
)


def get_names(
    lines: Optional[Union[str, List[str]]] = None,
    mols: Optional[Union[str, List[str]]] = None,
) -> Tuple[List[str], List[str]]:
    """TODO"""

    # Loading
    df_train: pd.DataFrame = pd.read_pickle(DATA_PATH.format("train"))

    # Features names
    inputs_names = df_train.columns.to_list()[:4]
    outputs_names = _select_lines(df_train.columns.to_list()[4:], lines, mols)

    return inputs_names, outputs_names


def load_data_pandas(
    lines: Optional[Union[str, List[str]]] = None,
    mols: Optional[Union[str, List[str]]] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # Loading
    df_train: pd.DataFrame = pd.read_pickle(DATA_PATH.format("train"))
    df_val: pd.DataFrame = pd.read_pickle(DATA_PATH.format("test"))

    df_mask_train: pd.DataFrame = pd.read_pickle(DATA_PATH.format("train_mask"))
    df_mask_val: pd.DataFrame = pd.read_pickle(DATA_PATH.format("test_mask"))

    # Drop rows with non-physic values
    idx = df_train.iloc[:, :3].gt(0).all(1).values
    df_train = df_train[idx]
    df_mask_train = df_mask_train[idx]

    idx = df_train.iloc[:, 4:].gt(0).all(1).values
    df_train = df_train[idx]
    df_mask_train = df_mask_train[idx]

    idx = df_val.iloc[:, :3].gt(0).all(1).values
    df_val = df_val[idx]
    df_mask_val = df_mask_val[idx]

    idx = df_val.iloc[:, 4:].gt(0).all(1).values
    df_val = df_val[idx]
    df_mask_val = df_mask_val[idx]

    # Inputs and outputs splitting
    df_inputs_train = df_train.iloc[:, :4]
    df_outputs_train = df_train.iloc[:, 4:]

    df_inputs_val = df_val.iloc[:, :4]
    df_outputs_val = df_val.iloc[:, 4:]

    df_mask_train = df_mask_train.iloc[:, 4:]
    df_mask_val = df_mask_val.iloc[:, 4:]

    selected_outputs = _select_lines(df_outputs_train.columns.to_list(), lines, mols)

    # Conversion to log
    df_outputs_train = df_outputs_train.apply(np.log10)
    df_outputs_val = df_outputs_val.apply(np.log10)

    n_inputs = 4
    n_outputs = len(selected_outputs)

    print(f"\nNumber of input features: {n_inputs:,}")
    print(f"Number of output features: {n_outputs:,}")
    print(f"Number of rows: {len(df_inputs_train):,}\n")

    # Selecting columns and conversion to arrays
    df_outputs_train = df_outputs_train[selected_outputs]
    df_outputs_val = df_outputs_val[selected_outputs]
    df_mask_train = df_mask_train[selected_outputs]
    df_mask_val = df_mask_val[selected_outputs]

    return (
        df_inputs_train,
        df_outputs_train,
        df_inputs_val,
        df_outputs_val,
        df_mask_train,
        df_mask_val,
    )


def prepare_data(
    lines: Optional[Union[str, List[str]]] = None,
    mols: Optional[Union[str, List[str]]] = None,
) -> Tuple[RegressionDataset, RegressionDataset, MaskDataset, MaskDataset]:
    """TODO"""
    (
        df_inputs_train,
        df_outputs_train,
        df_inputs_val,
        df_outputs_val,
        df_mask_train,
        df_mask_val,
    ) = load_data_pandas(lines, mols)

    # Conversion to log and float32
    df_inputs_train = df_inputs_train.astype("float32")
    df_outputs_train = df_outputs_train.astype("float32")
    df_inputs_val = df_inputs_val.astype("float32")
    df_outputs_val = df_outputs_val.astype("float32")

    # Dataset creation
    dataset_train = RegressionDataset.from_pandas(
        df_inputs_train,
        df_outputs_train,
    )
    dataset_val = RegressionDataset.from_pandas(
        df_inputs_val,
        df_outputs_val,
    )
    dataset_mask_train = MaskDataset.from_pandas(
        1 - df_mask_train,
    )
    dataset_mask_val = MaskDataset.from_pandas(
        1 - df_mask_val,
    )

    return dataset_train, dataset_val, dataset_mask_train, dataset_mask_val


def prepare_clusters(
    n_clusters: int,
    lines: Optional[Union[str, List[str]]] = None,
    mols: Optional[Union[str, List[str]]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """TODO"""

    path = os.path.join(os.path.dirname(__file__), "..", "out-clustering", "{}")

    with open(path.format(f"{n_clusters}_clusters.json"), "r") as f:
        clusters = json.load(f)

    # Change string keys into integers
    clusters = dict((int(key), value) for (key, value) in clusters.items())

    # Filter lines
    for k in clusters:
        if lines is not None:
            clusters[k] = [line for line in clusters[k] if line in lines]
        elif mols is not None:
            clusters[k] = filter_molecules(clusters[k], mols)

    return clusters


def _select_lines(
    available_lines: List[str],
    lines: Optional[Union[str, List[str]]] = None,
    mols: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """TODO"""

    if lines is not None and mols is not None:
        raise ValueError("lines and mols cannot be specified simultaneously")

    # Lines selection
    outputs_to_drop = [
        "so_n0_j1__n1_j0",  # Population inversion cause instabilities
        "h_el3s_j1_2__el2p_j3_2",  # Atomic lines with wavelengths lower than 6700 Angstrom
        "h_el3d_j3_2__el2p_j3_2",  # ...
        "h_el3d_j5_2__el2p_j3_2",
        "h_el3p_j1_2__el2s_j1_2",
        "h_el3s_j1_2__el2p_j1_2",
        "h_el3p_j3_2__el2s_j1_2",
        "h_el3d_j3_2__el2p_j1_2",
        "h_el2p_j1_2__el1s_j1_2",
        "h_el2s_j1_2__el1s_j1_2",
        "h_el2p_j3_2__el1s_j1_2",
        "h_el3p_j1_2__el1s_j1_2",
        "h_el3s_j1_2__el1s_j1_2",
        "h_el3d_j3_2__el1s_j1_2",
        "h_el3p_j3_2__el1s_j1_2",
        "h_el3d_j5_2__el1s_j1_2",
        "c_el1s_j0__el3p_j2",
        "c_el1s_j0__el3p_j1",
        "n_el2do_j5_2__el4so_j3_2",
        "n_el2do_j3_2__el4so_j3_2",
        "n_el2po_j1_2__el4so_j3_2",
        "n_el2po_j3_2__el4so_j3_2",
        "o_el1d_j2__el3p_j0",
        "o_el1d_j2__el3p_j1",
        "o_el1d_j2__el3p_j2",
        "o_el1s_j0__el1d_j2",
        "o_el1s_j0__el3p_j1",
        "o_el1s_j0__el3p_j2",
        "s_el1s_j0__el3p_j1",
        "s_el1s_j0__el3p_j2",
        "si_el1s_j0__el3p_j2",
        "si_el1s_j0__el3p_j1",
        "sp_el2po_j1_2__el4so_j3_2",
        "sp_el2po_j3_2__el4so_j3_2",
    ]  # Can be modified

    selected_outputs: List[str] = [
        line for line in available_lines if not line in outputs_to_drop
    ]

    if mols is not None:
        selected_outputs = filter_molecules(selected_outputs, mols)
    if lines is not None:
        selected_outputs = [line for line in lines if line in selected_outputs]

    return selected_outputs


def normalization_operators(
    df: pd.DataFrame, norm_type: NormTypes
) -> Tuple[Normalizer, InverseNormalizer]:
    return Normalizer(df, norm_type), InverseNormalizer(df, norm_type)


def build_data_transformers(
    dataset_train: RegressionDataset,
) -> Tuple[Operator, Operator, Operator, Operator]:
    # x operators

    scale_operator_x = ColumnwiseOperator(
        [
            log10,  # P
            log10,  # radm
            log10,  # Avmax
            id,  # angle
        ]
    )

    norm_type = NormTypes.MEAN0STD1

    norm_operator_x, unnorm_operator_x = normalization_operators(
        dataset_train.apply_transf(scale_operator_x, None).to_pandas()[0], norm_type
    )

    unscale_operator_x = ColumnwiseOperator(
        [
            pow10,  # P
            pow10,  # radm
            pow10,  # Avmax
            id,  # angle
        ]
    )

    operator_x = SequentialOperator([scale_operator_x, norm_operator_x])
    inverse_operator_x = SequentialOperator([unnorm_operator_x, unscale_operator_x])

    # y operators

    # operator_y = Operator(log10)
    # inverse_operator_y = Operator(pow10)

    operator_y = Operator(id)
    inverse_operator_y = Operator(id)

    return operator_x, inverse_operator_x, operator_y, inverse_operator_y
