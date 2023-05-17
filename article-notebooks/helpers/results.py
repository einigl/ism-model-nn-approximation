import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from nnbma.dataset import MaskDataset, RegressionDataset
from nnbma.learning import LearningParameters, learning_procedure
from nnbma.networks import NeuralNetwork
from nnbma.preprocessing import (
    ColumnwiseOperator,
    NormTypes,
    Operator,
    SequentialOperator,
    id,
    log10,
    pow10,
)

from helpers.preprocessing import prepare_data, normalization_operators
from helpers.plots import Plotter
from helpers.lines import filter_molecules, molecules_among_lines, molecule_and_transition


OUT_PATH = os.path.join(os.path.dirname(__file__), "nn-regression-results")


def save_readme(
    model: NeuralNetwork,
    learning_params: Union[LearningParameters, List[LearningParameters]],
    duration: datetime.timedelta,
    path: str,
) -> None:
    """TODO"""

    filename = os.path.join(path, "README.txt")

    s = duration.seconds + 24 * 3600 * duration.days
    h = s // 3600
    s -= h
    m = s // 60
    s -= m

    n_batch = 1_000
    repeat = 50
    _mean, _min, _max = model.time(n_batch, repeat)
    _pm = max(_mean-_min, _max-_mean)

    with open(filename, "wt") as f:
        f.writelines(
            [
                "README\n",
                "------\n\n",
                f"Date: {datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S')}\n",
                f"Duration: {h:02d}:{m:02d}:{s:02d}\n\n",
                f"Learning parameters: {learning_params if isinstance(learning_params, LearningParameters) else [str(lp) for lp in learning_params]}\n\n",
                f"Network: {model}\n\n",
                f"Number of learnable parameters: {model.count_parameters():,}\n",
                f"Size of learnable parameters: {model.count_bytes(display=True)}\n\n",
                f"Computation time ({n_batch} entries, {repeat} iterations): {1e3*_mean:.3f} ms +/- {1e3*_pm:.3f}"
            ]
        )

    return


def save_losses(
    results: Dict[str, Any],
    path: str,
) -> None:
    """TODO"""

    filename = os.path.join(path, "losses.png")

    plt.rc("text", usetex=True)

    plt.figure(dpi=125)

    plt.semilogy(results["train_loss"], color="blue", label="Training")
    plt.semilogy(results["val_loss"], color="red", label="Validation")

    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    plt.rc("text", usetex=False)


def save_relative_errors(
    results: Dict[str, Any],
    path: str,
) -> None:
    """TODO"""

    filename = os.path.join(path, "errors.png")

    plt.figure(dpi=125)

    plt.semilogy(results["train_relerr"], "b-", label="Training (mean)")
    plt.semilogy(results["val_relerr"], "r-", label="Validation (mean)")

    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Error factor")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # 

    filename = os.path.join(path, "errors_asymptotic.png")

    plt.figure(dpi=125)

    plt.semilogy(results["train_relerr"], "b-", label="Training (mean)")
    plt.semilogy(results["val_relerr"], "r-", label="Validation (mean)")
    plt.ylim([None, 10])

    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Error factor")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_lr(
    results: Dict[str, Any],
    path: str,
) -> None:
    """TODO"""

    filename = os.path.join(path, "lr.png")

    plt.figure(dpi=125)

    plt.semilogy(results["lr"], color="blue", label="Learning rate")

    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_batch_size(
    results: Dict[str, Any],
    path: str,
) -> None:
    """TODO"""

    filename = os.path.join(path, "batch_size.png")

    plt.figure(dpi=125)

    plt.semilogy(results["batch_size"], color="blue", label="Batch size")

    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Batch size")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compute_errors(
    dataset_ref: RegressionDataset,
    dataset_out: RegressionDataset,
    path: str,
    dirname: str,
    dataset_mask: Optional[MaskDataset] = None,
):
    """TODO"""

    os.mkdir(os.path.join(path, dirname))

    y_ref = dataset_ref.getall(numpy=True)[1]
    y_out = dataset_out.getall(numpy=True)[1]

    rel_err = 100 * (10 ** np.abs(y_out - y_ref) - 1)
    if dataset_mask is not None:
        mask = dataset_mask.getall(numpy=True)
        rel_err = np.where(mask, rel_err, np.nan)

    rows = ["mean", "min", "median", "90%", "95%", "99%", "max"]

    # All lines

    _perc = np.nanpercentile(rel_err, (0, 50, 90, 95, 99, 100), axis=0)
    data = np.row_stack(
        (
            np.nanmean(rel_err, axis=0),
            _perc[0],
            _perc[1],
            _perc[2],
            _perc[3],
            _perc[4],
            _perc[5],
        )
    )

    df = pd.DataFrame(data, index=rows, columns=dataset_ref.outputs_names)

    # Each molecules

    molecules = molecules_among_lines(dataset_ref.outputs_names)

    data_summary = np.empty((len(df.index), 0))
    for mol in molecules:
        lines_mol = filter_molecules(dataset_ref.outputs_names, mol)
        df_mol = df[lines_mol]
        rel_err_mol = rel_err[
            :, [dataset_ref.outputs_names.index(name) for name in lines_mol]
        ]

        _perc = np.nanpercentile(rel_err_mol, (0, 50, 90, 95, 99, 100))
        _data = np.array(
            (
                np.nanmean(rel_err_mol),
                _perc[0],
                _perc[1],
                _perc[2],
                _perc[3],
                _perc[4],
                _perc[5],
            )
        )

        filename = os.path.join(path, dirname, f"{dirname}_{mol}.csv")

        df_mol.insert(0, "all", _data)
        df_mol.to_csv(filename)

        data_summary = np.column_stack((data_summary, _data))

    filename = os.path.join(path, f"{dirname}_summary.csv")

    _perc = np.nanpercentile(rel_err, (0, 50, 90, 95, 99, 100))
    data_all = np.row_stack(
        (
            np.nanmean(rel_err),
            _perc[0],
            _perc[1],
            _perc[2],
            _perc[3],
            _perc[4],
            _perc[5],
        )
    )

    df_summary = pd.DataFrame(data_summary, index=rows, columns=molecules)
    df_summary.insert(0, "all", data_all)
    df_summary.to_csv(filename)


def badly_reconstructed(
    n_points: int,
    dataset_ref: RegressionDataset,
    dataset_out: RegressionDataset,
    path: str,
    filename: str,
    dataset_mask: Optional[MaskDataset] = None,
    model: Optional[NeuralNetwork] = None,
    n_profiles: Optional[int] = None,
):
    """TODO"""

    filename = os.path.join(path, os.path.splitext(filename)[0] + ".csv")

    x, y_ref = dataset_ref.getall(numpy=True)
    _, y_out = dataset_out.getall(numpy=True)
    if dataset_mask is not None:
        mask = dataset_mask.getall(numpy=True)

    # Select only rows whose angle is 0
    indices = x[:, 3] == 0
    x = x[indices]
    y_ref = y_ref[indices]
    y_out = y_out[indices]
    if dataset_mask is not None:
        mask = mask[indices]

    errors = np.abs(y_out - y_ref)
    if dataset_mask is not None:
        errors = np.where(mask, errors, np.nan)
    rel_err = 100 * (10**errors - 1)

    indices = np.argsort(
        np.where(np.isnan(errors.flatten()), -np.inf, errors.flatten())
    )[::-1]
    rows, cols = np.unravel_index(indices, errors.shape)

    data = [
        [dataset_ref.outputs_names[cols[n]]]\
            + list(x[rows[n]])\
            + [rel_err[rows[n], cols[n]], errors[rows[n], cols[n]], y_ref[rows[n], cols[n]], y_out[rows[n], cols[n]]]\
        for n in tqdm(range(round(0.01 * rows.shape[0])))]
            
    df = pd.DataFrame(
        data,
        columns=["line"]
        + dataset_ref.inputs_names
        + ["Error factor (%)", "Abs. error (log)", "Target value", "Estimated value"]
    )

    df.iloc[:n_points].to_csv(filename)

    if n_profiles is None:
        return

    # Profiles plotting

    if model is None:
        return
    
    plotter = Plotter(
        df_inputs=dataset_ref.to_pandas()[0],
        df_outputs=dataset_ref.to_pandas()[1],
        df_mask=dataset_mask.to_pandas() if dataset_mask is not None else None,
        model=model,
    )

    df = df[dataset_ref.inputs_names + ["line"]].rename(columns={"line": "lines"})

    all_lines = dataset_out.outputs_names
    new_df = pd.DataFrame(columns=df.columns)
    for _ in tqdm(range(n_profiles)):
        worst = df.iloc[0]
        mol_lines = filter_molecules(all_lines, molecule_and_transition(worst['lines'])[0])

        rows = df[(df['P']==worst['P']) & (df['radm']==worst['radm']) & (df['Avmax']==worst['Avmax']) & (df['lines'].isin(mol_lines))]
        new_df = pd.concat([new_df, df.iloc[:1]])
        df = df.drop(index=rows.index)
        if len(df) == 0:
            break

    new_df = new_df.reset_index()
    plotter.save_profiles_from_csv(
        new_df,
        path_outputs=os.path.join(path, "badly_reconstructed_profiles"),
        grid=True,
        regression=True,
        errors=True,
    )
    new_df.to_csv(os.path.join(path, "badly_reconstructed_profiles", "table.csv"))


def errors_distributions(
    dataset_ref: RegressionDataset,
    dataset_out: RegressionDataset,
    path: str,
    dirname: str,
    dataset_mask: Optional[MaskDataset] = None,
):
    """TODO"""

    path = os.path.join(path, dirname)

    os.mkdir(path)

    # Errors computation

    x, y_ref = dataset_ref.getall(numpy=True)
    _, y_out = dataset_out.getall(numpy=True)
    if dataset_mask is not None:
        mask = dataset_mask.getall(numpy=True)
        y_ref = np.where(mask, y_ref, np.nan)
        y_out = np.where(mask, y_out, np.nan)

    errors = np.abs(y_out - y_ref)

    # Percentiles of errors versus target value

    filename = os.path.join(path, "percentiles.png")

    win_factor = 10
    n_points = 100

    points = np.logspace(
        np.log10(win_factor) + np.nanmin(y_ref),
        np.nanmax(y_ref) - np.log10(win_factor),
        n_points,
    )

    means = np.zeros(n_points)
    percs = np.zeros((n_points, 5))

    for i, p in enumerate(points):
        _data = errors[(y_ref > p / win_factor) & (y_ref < p * win_factor)]
        means[i] = np.nanmean(_data)
        percs[i] = np.nanpercentile(_data, (0.5, 5, 50, 95, 99.5))

    plt.figure(dpi=125)

    plt.plot(points, means, label="Mean")
    plt.plot(points, percs[:, 2], label="Median")
    plt.fill_between(points, percs[:, 0], percs[:, 4], label="99\\%", alpha=0.5)
    plt.fill_between(points, percs[:, 1], percs[:, 3], label="90\\%", alpha=0.5)
    plt.xscale("log")

    plt.xlabel("Target value")
    plt.ylabel("Abs. error (log)")
    plt.legend()

    plt.savefig(filename)
    plt.close()


def masked_values(
    dataset: RegressionDataset,
    dataset_mask: MaskDataset,
    path: str,
    filename: str,
) -> None:
    """TODO"""
    filename = os.path.join(path, os.path.splitext(filename)[0] + ".csv")

    df_inputs, _ = dataset.to_pandas()
    df_mask = dataset_mask.to_pandas()
    df_out = pd.DataFrame(data=[], columns=df_inputs.columns.to_list() + ["line"])
    k = 1
    for n in df_mask.index:
        masked_lines = df_mask.columns[df_mask.iloc[n] == 0].to_list()
        for line in masked_lines:
            df_out.loc[k] = df_inputs.loc[n].to_list() + [line]
            k += 1
    df_out.to_csv(filename, index=False)
    return


### FULL PROCEDURE ###
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


def procedure(
    lines: List[str],
    model: NeuralNetwork,
    learning_params: LearningParameters,
    dirname: str,
    mask: bool=True,
    architecture_name: Optional[str]=None,
    plot_profiles: bool=True,
    verbose: bool=True,
) -> None:
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

    # meth = getattr(model, "update_standardization", None)
    # if callable(meth):
    #     meth(dataset_train_transf.getall()[0]) TODO

    ## Learning procedure

    results = learning_procedure(
        model,
        (dataset_train_transf, dataset_val_transf),
        learning_params,
        (dataset_mask_train, dataset_mask_val),
        verbose=verbose,
    )

    ## Get predictions

    _x = dataset_train.getall(numpy=True)[0]
    _y = model.evaluate(_x, True, True)

    dataset_train_out = RegressionDataset(
        _x,
        _y,
        dataset_train.inputs_names,
        dataset_train.outputs_names,
    )

    _x = dataset_val.getall(numpy=True)[0]
    _y = model.evaluate(_x, True, True)

    dataset_val_out = RegressionDataset(
        _x,
        _y,
        dataset_val.inputs_names,
        dataset_val.outputs_names,
    )

    ## Save results

    # Creates directory
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)

    path = os.path.join(OUT_PATH, dirname)
    if not os.path.isdir(path):
        os.mkdir(path)

    if architecture_name is None:
        path = os.path.join(path, datetime.datetime.now().strftime("%Y_%m_%d"))
    else:
        path = os.path.join(path, architecture_name)

    if os.path.isdir(path):
        path += "__{}"
        i = 1
        while os.path.isdir(path.format(i)):
            i += 1
        path = path.format(i)
    os.mkdir(path)

    # Save README.txt

    save_readme(model, learning_params, results["duration"], path)
    print("README saved")

    # Save learning

    save_losses(results, path)
    save_relative_errors(results, path)
    save_lr(results, path)
    save_batch_size(results, path)
    print("Learning figures saved")

    # Save model

    model.save("architecture", path)
    print("Architecture saved")

    # Save performances

    compute_errors(
        dataset_train,
        dataset_train_out,
        path,
        "errors_train",
        dataset_mask_train,
    )

    compute_errors(
        dataset_val,
        dataset_val_out,
        path,
        "errors_val",
        dataset_mask_val,
    )

    print("Spreadsheets of errors saved")

    # Save the most badly reconstructed points

    badly_reconstructed(
        10_000,
        dataset_train,
        dataset_train_out,
        path,
        "badly_reconstructed_train",
        dataset_mask_train,
        model=model,
        n_profiles=20 if plot_profiles else None,
    )

    badly_reconstructed(
        10_000,
        dataset_val,
        dataset_val_out,
        path,
        "badly_reconstructed_val",
        dataset_mask_val,
    )

    print("Spreadsheet of worst estimations and their profiles saved")

    # Save some graphical representation of the errors distribution

    # errors_distributions(
    #     dataset_train,
    #     dataset_train_out,
    #     path,
    #     "errors_distributions_train",
    #     dataset_mask_train
    # )

    # errors_distributions(
    #     dataset_val,
    #     dataset_val_out,
    #     path,
    #     "errors_distributions_val",
    #     dataset_mask_val
    # )

    # print('Errors distributions figures saved')

    # Save masked values spreadsheet
    if dataset_mask_train is not None:
        masked_values(dataset_train, dataset_mask_train, path, "masked_values_train")

    if dataset_mask_val is not None:
        masked_values(dataset_val, dataset_mask_val, path, "masked_values_val")

    print("Spreadsheet of masked values saved")
