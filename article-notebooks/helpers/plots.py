import os
import shutil
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm

from nnbma.networks import NeuralNetwork

from .lines import line_to_latex

__all__ = ["LaTeX", "Plotter"]


class LaTeX:
    """
    Class to handle activation of the plotting with latex if available on the current installation.
    
    Example:
    ```
    with LaTeX():
        # Do some matplotlib stuff
    ```
    """
    activate: bool
    previous_mode: bool

    def __init__(self, activate: bool = True):
        if not isinstance(activate, bool):
            raise TypeError(f"activate must be a boolean value, not {type(activate)}")
        self.activate = activate
        self.previous_mode = plt.rcParams["text.usetex"]

    def __enter__(self):
        if self.activate:
            plt.rc("text", usetex=shutil.which("latex") is not None)
        else:
            plt.rc("text", usetex=False)
        return self

    def __exit__(self, _, __, ___):
        plt.rc("text", usetex=self.previous_mode)


class Plotter:
    """
    Class to handle the plotting of profiles and slices in the parameter space in a very user-friendly way.
    """

    _df: pd.DataFrame
    _df_mask: pd.DataFrame
    _df_mask: Optional[pd.DataFrame]
    _model: Optional[NeuralNetwork]
    _grid: Dict[str, List[float]]
    _inputs_names: List[str]
    _inputs_latex: List[str]
    _inputs_units: List[str]
    _inputs_units_long: List[str]
    _inputs_scales: List[str]
    _outputs_names: List[str]

    def __init__(
        self,
        df_inputs: pd.DataFrame,
        df_outputs: pd.DataFrame,
        df_mask: Optional[pd.DataFrame] = None,
        model: Optional[NeuralNetwork] = None,
    ):
        """TODO"""
        self._model = model

        if df_mask is None:
            df_mask = pd.DataFrame(
                0, index=df_outputs.index, columns=df_outputs.columns
            )
        elif df_mask.mean().mean() > 0.5:
            df_mask = 1 - df_mask

        self._df = pd.concat([df_inputs, df_outputs], axis=1)
        self._df_mask = pd.concat([df_inputs, df_mask], axis=1)

        self._grid = {}
        for param in df_inputs.columns.to_list():
            param: str
            values = df_inputs[param].drop_duplicates().to_list()
            values.sort()
            self._grid.update({param: values})

        inputs_names = ["P", "Avmax", "radm", "angle"]
        inputs_latex = ["P", "Av^{tot}", "G_0", "\\alpha"]
        inputs_units = ["K.cm$^{-3}$", "mag", "", "deg"]
        inputs_units_long = ["K.cm$^{-3}$", "mag", "Mathis units", "deg"]
        inputs_scales = ["log", "log", "log", "lin"]

        indices = []
        for param in inputs_names:
            if not param in df_inputs.columns:
                raise ValueError(
                    f"Parameters {df_inputs.columns} are incompatible with {inputs_names}"
                )
            indices.append(df_inputs.columns.to_list().index(param))

        self._inputs_names = [inputs_names[i] for i in indices]
        self._inputs_latex = [inputs_latex[i] for i in indices]
        self._inputs_units = [inputs_units[i] for i in indices]
        self._inputs_units_long = [inputs_units_long[i] for i in indices]
        self._inputs_scales = [inputs_scales[i] for i in indices]

        self._outputs_names = df_outputs.columns.to_list()

    @property
    def inputs_names(self):
        """TODO"""
        return self._inputs_names

    @property
    def outputs_names(self):
        """TODO"""
        return self._outputs_names

    @property
    def n_inputs(self):
        """TODO"""
        return len(self._inputs_names)

    @property
    def n_outputs(self):
        """TODO"""
        return len(self._outputs_names)

    @property
    def grid(self):
        """TODO"""
        return self._grid

    def print_grid(self) -> None:
        """TODO"""
        print("Parameters grid")
        for param in self.grid:
            ls = [f"{val:.2e}" for val in self.grid[param]]
            print(f"{param}:", *ls)

    def print_grid_shape(self) -> None:
        """TODO"""
        print("Parameters grid")
        for param in self.grid:
            print(f"{param}:", len(self.grid[param]))

    def closest_in_grid(
        self,
        P: Optional[float] = None,
        radm: Optional[float] = None,
        Avmax: Optional[float] = None,
        angle: Optional[float] = None,
    ) -> Dict[str, float]:
        res = {}
        for param, scale in zip(self._inputs_names, self._inputs_scales):
            value = locals()[param]
            if value is not None:
                if scale == "log":
                    closest = min(
                        self.grid[param], key=lambda x: abs(np.log(x) - np.log(value))
                    )
                else:
                    closest = min(self.grid[param], key=lambda x: abs(x - value))
                res.update({param: closest})
        return res

    def _check_args(
        self,
        n_samples: int,
        grid: Optional[bool],
        regression: Optional[bool],
        errors: bool,
        legend: bool,
        latex: bool,
    ) -> None:
        """ """
        if not isinstance(n_samples, int):
            raise TypeError(f"n_samples must be an int, not {type(n_samples)}")

        if not isinstance(grid, bool) and grid is not None:
            raise TypeError(f"grid must be a bool, not {type(grid)}")
        if not isinstance(regression, bool) and regression is not None:
            raise TypeError(
                f"regression must be a bool or None, not {type(regression)}"
            )
        if not isinstance(errors, bool):
            raise TypeError(f"errors must be a bool, not {type(errors)}")

        if not isinstance(legend, bool):
            raise TypeError(f"legend must be a bool, not {type(legend)}")
        if not isinstance(latex, bool):
            raise TypeError(f"latex must be a bool, not {type(latex)}")

    def _parse_csv(
            self,
            csv_file: Union[str, pd.DataFrame]
        ) -> pd.DataFrame:
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            df = csv_file
        else:
            raise TypeError(
                f"csv_file must be a string path or a DataFrame, not {type(csv_file)}"
            )
        # Check the columns names
        inputs = [name.strip().lower() for name in self.inputs_names + ["lines"]]
        columns = [name.strip().lower() for name in df.columns]
        try:
            indices = [columns.index(name) for name in inputs]
        except ValueError:
            raise ValueError(
                f"Columns of loaded file does not match {inputs} even rearranging them."
            )
        columns = [df.columns[i] for i in indices]
        df = df[columns]
        df = df.rename(
            columns={
                name_low: name for name_low, name in zip(inputs[-1], self.inputs_names)
            }
        )
        return df

    def _plot_profile(
        self,
        lines_to_plot: List[str],
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        y_grid: Optional[np.ndarray],
        values: List[float],
        df: pd.DataFrame,
        df_mask: pd.DataFrame,
        k_none: int,
        grid: bool,
        regression: bool,
        errors: bool,
        highlighted_indices: List[int],
        legend: bool,
        fontsize: int,
    ):
        """TODO"""
        # Argument checking
        if (grid or regression) and errors:
            raise ValueError(
                "Grid or regression must not be True simultaneously with errors"
            )
        
        # Plot profiles
        x_op = lambda t: t
        y_op = lambda t: 10**t
        for idx, line in enumerate(lines_to_plot):

            m = df_mask[line].values.astype(np.bool)

            if errors:
                err = 100 * (10 ** np.abs(y_grid[:, idx] - df[line].values) - 1)
                lines = plt.plot(
                    x_op(df.iloc[:, k_none].values[~m]),
                    err[~m],
                    linestyle="--",
                    marker="x",
                    markersize=10,
                    label=line_to_latex(line),
                )
                plt.scatter(
                    x_op(df.iloc[:, k_none].values[highlighted_indices]),
                    err[highlighted_indices],
                    s=10**2,
                    marker="x",
                    linewidth=3,
                    color=lines[0].get_color(),
                )

                plt.ylabel("Errors factor (\%)", labelpad=15, fontsize=fontsize)

            else:
                lines = None

                if grid:
                    lines = plt.semilogy(
                        x_op(df.iloc[:, k_none]),
                        y_op(df[line].values),
                        linestyle="--",
                        marker="x",
                        markersize=10,
                        label=None if regression else line_to_latex(line),
                    )
                    plt.scatter(
                        x_op(df.iloc[:, k_none].values[m]),
                        y_op(df[line].values[m]),
                        s=11**2,
                        marker="s",
                        color=lines[0].get_color(),
                    )

                if regression:
                    lines = plt.semilogy(
                        x_op(X[:, k_none]), y_op(y[:, idx]),
                        label=line_to_latex(line),
                        color=lines[0].get_color() if grid else None
                    )

                plt.scatter(
                    x_op(df.iloc[:, k_none].values[highlighted_indices]),
                    y_op(df[line].values[highlighted_indices]),
                    s=10**2,
                    marker="x",
                    linewidth=3,
                    color=lines[0].get_color(),
                )

                plt.ylabel(
                    "Integrated intensities",
                    # "Integrated intensities\n(erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)",
                    labelpad=15,
                    fontsize=fontsize,
                )

        rg_min = 10
        if errors is not True:
            if regression is True:
                mini = y_op(min(y.min(), df[lines_to_plot].values.min()))
                maxi = y_op(max(y.max(), df[lines_to_plot].values.max()))
            else:
                mini = y_op(df[lines_to_plot].values.min())
                maxi = y_op(df[lines_to_plot].values.max())
            rg = maxi / mini
            if rg < rg_min:  # Can be modified
                plt.ylim([mini / (rg_min / rg) ** 0.5, maxi * (rg_min / rg) ** 0.5])

        if self._inputs_scales[k_none] == "log":
            plt.xscale("log")

        plt.grid()
        plt.xlabel(
            f"${self._inputs_latex[k_none]}$ ({self._inputs_units_long[k_none]})".replace(
                "()", "(-)"
            ),
            labelpad=15,
            fontsize=fontsize,
        )
        str_title = ""
        for k in range(self.n_inputs):
            if k == k_none:
                continue
            str_title += (
                "${}={:.1e}$ {}, "
                if self._inputs_scales[k] == "log"
                else "${}={:.2f}$ {}, "
            ).format(self._inputs_latex[k], values[k], self._inputs_units[k])
        str_title = str_title.replace(" , ", ", ").removesuffix(", ")
        plt.title(str_title, pad=15, fontsize=int(fontsize * 1.2))
        if legend:
            plt.legend(
                fontsize=fontsize,
            )

        plt.gca().tick_params(axis="both", labelsize=int(fontsize * 1.2))

    def plot_profile(
        self,
        lines_to_plot: Union[List[str], str],
        P: Optional[float] = None,
        radm: Optional[float] = None,
        Avmax: Optional[float] = None,
        angle: Optional[float] = None,
        n_samples: int = 100,
        grid: Optional[bool] = None,
        regression: Optional[bool] = None,
        errors: bool = False,
        highlighted: List[float] = [],
        legend: bool = True,
        latex: bool = True,
        fontsize: int = 10,
    ):
        """
        Only one variable among P, Avmax, radm and angle has to be null.
        """
        # Arguments checking
        self._check_args(n_samples, grid, regression, errors, legend, latex)

        if isinstance(lines_to_plot, str):
            lines_to_plot = [lines_to_plot]
        elif not isinstance(lines_to_plot, list):
            raise TypeError(f"lines_to_plot must be a list, not {type(lines_to_plot)}")
        elif any([not isinstance(line, str) for line in lines_to_plot]):
            raise TypeError("lines_to_plot must be a list of str")

        if regression and self._model is None:
            raise ValueError("regression must not be True if no model has been given")
        if errors:
            grid = grid if grid is not None else False
            regression = regression if regression is not None else False
        else:
            grid = grid if grid is not None else True
            regression = (
                regression if regression is not None else (self._model is not None)
            )

        if isinstance(highlighted, (float, int, np.floating)):
            highlighted = [highlighted]
        elif not isinstance(highlighted, list):
            raise TypeError(
                f"highlighted must be a list or a float, not {type(highlighted)}"
            )

        # DataFrames
        # df = pd.concat([self._df_inputs, self._df_outputs], axis=1)
        # df_mask = pd.concat([self._df_inputs, self._df_mask], axis=1)

        ks_none = []
        for k in range(self.n_inputs):
            value = locals()[self.inputs_names[k]]
            if value is None:
                ks_none.append(k)

        if len(ks_none) != 1:
            raise ValueError("The number of None inputs is different from 1.")
        k_none = ks_none[0]
        param_none = self._inputs_names[k_none]

        # Real lines profiles
        values = []
        list_params_filter = []
        list_closest_filter = []
        for param in self.inputs_names:
            value = locals()[param]
            if value is not None:
                closest = self.closest_in_grid(**{param: value})[param]
                values.append(closest)
                list_params_filter.append(param)
                list_closest_filter.append(closest)
            else:
                values.append(None)

        assert len(list_params_filter) == 3
        df = self._df.loc[
            (self._df[list_params_filter[0]] == list_closest_filter[0])
            & (self._df[list_params_filter[1]] == list_closest_filter[1])
            & (self._df[list_params_filter[2]] == list_closest_filter[2]),
            self._inputs_names + lines_to_plot,
        ]

        df_mask = self._df_mask.loc[
            (self._df_mask[list_params_filter[0]] == list_closest_filter[0])
            & (self._df_mask[list_params_filter[1]] == list_closest_filter[1])
            & (self._df_mask[list_params_filter[2]] == list_closest_filter[2]),
            self._inputs_names + lines_to_plot,
        ]

        X_grid = np.zeros((len(df), self.n_inputs))
        X = np.zeros((n_samples, self.n_inputs))

        for k in range(self.n_inputs):
            value = values[k]
            if value is None:
                if self._inputs_scales[k] == "log":
                    X[:, k] = np.logspace(
                        np.log10(self.grid[self.inputs_names[k]][0]),
                        np.log10(self.grid[self.inputs_names[k]][-1]),
                        n_samples,
                    )
                    
                else:
                    X[:, k] = np.linspace(
                        self.grid[self.inputs_names[k]][0],
                        self.grid[self.inputs_names[k]][-1],
                        n_samples,
                    )
                X_grid[:, k] = df[self.inputs_names[k]].values
            else:
                X[:, k] = value
                X_grid[:, k] = value

        # Neural network approximation
        if regression or errors:
            # Evaluate model
            previous_output_subset = self._model.current_output_subset
            self._model.eval()
            self._model.restrict_to_output_subset(
                lines_to_plot
            )  # Restrict only to line we want
            y = self._model.evaluate(X, transform_inputs=True, transform_outputs=True)
            y_grid = self._model.evaluate(X_grid, transform_inputs=True, transform_outputs=True)
            self._model.restrict_to_output_subset(
                previous_output_subset
            )  # Restore the previous restriction
        else:
            y = None

        # Highlighted values
        if self._inputs_scales[k_none] == "log":
            highlighted_indices = [
                min(
                    range(len(df.iloc[:, k_none].values)),
                    key=lambda i: abs(np.log(df[param_none].values[i]) - np.log(h)),
                )
                for h in highlighted
            ]
        else:
            highlighted_indices = [
                min(
                    range(len(df.iloc[:, k_none].values)),
                    key=lambda i: abs(df[param_none].values[i] - h),
                )
                for h in highlighted
            ]

        # Plot profiles
        kwargs = {
            "lines_to_plot": lines_to_plot,
            "X": X if regression else None,
            "y": y if regression else None,
            "y_grid": y_grid if errors else None,
            "values": values,
            "df": df,
            "df_mask": df_mask,
            "k_none": k_none,
            "highlighted_indices": highlighted_indices,
            "legend": legend,
            "fontsize": fontsize,
        }

        if ((grid or regression) and not errors) or (
            not (grid or regression) and errors
        ):
            with LaTeX(activate=latex):
                self._plot_profile(
                    **kwargs,
                    grid=grid,
                    regression=regression,
                    errors=errors,
                )
        else:
            with LaTeX(activate=latex):
                plt.subplot(2, 1, 1)
                self._plot_profile(
                    **kwargs,
                    grid=grid,
                    regression=regression,
                    errors=False,
                )
                plt.tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
                plt.xlabel(None)

                plt.subplot(2, 1, 2)
                self._plot_profile(
                    **kwargs,
                    grid=False,
                    regression=False,
                    errors=True,
                )
                plt.title(None)

        plt.tight_layout()

    def save_profiles_from_csv(
        self,
        csv_file: Union[str, pd.DataFrame],
        path_outputs: str,
        n_samples: int = 100,
        grid: Optional[bool] = None,
        regression: Optional[bool] = None,
        errors: bool = False,
        legend: bool = True,
        latex: bool = True,
        dpi: int = 150,
    ) -> None:
        """TODO"""
        # Parse CSV or DataFrame
        df = self._parse_csv(csv_file)

        # Create directory
        if not os.path.isdir(path_outputs):
            os.mkdir(path_outputs)

        # Process each row
        for i in df.index:
            dirname = os.path.join(path_outputs, str(i))
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            row = df.loc[i]
            # Ignore a row if it has no line to plot
            if row.isnull()["lines"]:
                continue
            else:
                lines = [subs for subs in row["lines"].split(" ") if len(subs) > 0]
            # Ignore a row if more than one parameter is blank, an error is raised
            if (row[: len(self.inputs_names)].isnull()).sum() > 1:
                continue
            # If a row has exactly one blank value, we plot this profile
            if (row[: len(self.inputs_names)].isnull()).sum() == 1:
                names_blank = [row.index[row.isnull()]]
            # If a row has no blank value, we plot all the possible profiles
            else:
                names_blank = self.inputs_names
            # Save all programmed profiles
            for n_blank in names_blank:
                d = {
                    name: (row[name] if name != n_blank else None)
                    for name in self.inputs_names
                }

                fig = plt.figure(dpi=dpi)
                self.plot_profile(
                    lines_to_plot=lines,
                    **d,
                    n_samples=n_samples,
                    grid=grid,
                    regression=regression,
                    errors=errors,
                    highlighted=[row[n_blank]] if len(names_blank) > 1 else [],
                    legend=legend,
                    latex=latex,
                )
                fig.savefig(os.path.join(dirname, n_blank))
                plt.close(fig)

    def _plot_slice(
        self,
        line_to_plot: str,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        y_grid: Optional[np.ndarray],
        values: List[float],
        df: pd.DataFrame,
        df_mask: pd.DataFrame,
        k_none_1: int,
        k_none_2: int,
        grid: bool,
        regression: bool,
        errors: bool,
        highlighted_1: List[Tuple[float, bool]],
        highlighted_2: List[Tuple[float, bool]],
        legend: bool,
        fontsize: int,
        pointsize: int,
        cmap: str,
    ) -> Colorbar:

        if (grid and regression) or (grid and errors) or (regression and errors):
            raise ValueError(
                "Only one argument among 'grid', 'regression' and 'errors' must be True"
            )
        if not grid and not regression and not errors:
            raise ValueError(
                f"One argument among 'grid', 'regression' and 'errors' must be True"
            )

        # Plot profiles
        x_op = lambda t: t
        y_op = lambda t: 10**t

        m = df_mask[line_to_plot].values.astype(np.bool)

        if errors:
            err = 100 * (10 ** np.abs(y_grid[:, 0] - df[line_to_plot].values) - 1)
            im = plt.scatter(
                x_op(df.iloc[:, k_none_1].values[~m]),
                x_op(df.iloc[:, k_none_2].values[~m]),
                c=err[~m],
                s=pointsize,
                label=line_to_latex(line_to_plot),
            )

            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel(
                "Errors factor (\%)", rotation=-90, fontsize=fontsize, labelpad=20
            )

            vmin, vmax = err[~m & (err > 0)].min(), err[~m & (err > 0)].max()

        elif grid or regression:

            if grid:
                _y = y_op(df[line_to_plot].values)
                vmin = _y[_y > 0].min()
                vmax = _y[_y > 0].max()
                assert vmin > 0, f"vmin should be strictly positive: {vmin}"

                norm = LogNorm(vmin, vmax)

                plt.scatter(
                    x_op(df.iloc[:, k_none_1].values[m]),
                    x_op(df.iloc[:, k_none_2].values[m]),
                    c=_y[m],
                    marker="s",
                    norm=norm,
                    s=pointsize,
                    cmap=cmap,
                )

                im = plt.scatter(
                    x_op(df.iloc[:, k_none_1]),
                    x_op(df.iloc[:, k_none_2]),
                    c=_y,
                    label=line_to_latex(line_to_plot),
                    norm=norm,
                    s=pointsize,
                    cmap=cmap,
                )

            else:
                n_samples = round(X.shape[0] ** 0.5)
                x1 = X[:, k_none_1].reshape(n_samples, n_samples)
                x2 = X[:, k_none_2].reshape(n_samples, n_samples)
                _y = y_op(y[:, 0]).reshape(n_samples, n_samples)
                norm = LogNorm(vmin=_y.min(), vmax=_y.max())
                im = plt.pcolor(
                    x1,
                    x2,
                    _y,
                    norm=norm,
                    cmap=cmap,
                )

                plt.scatter([], [], label=line_to_latex(line_to_plot))

            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel(
                "Integrated intensities",  # (erg.cm$^{-2}$.s$^{-1}$.sr$^{-1}$)",
                rotation=-90,
                fontsize=fontsize,
                labelpad=20,
            )

        # Highlighted values

        param_none_1 = self._inputs_names[k_none_1]
        param_none_2 = self._inputs_names[k_none_2]

        _idx = [
            (
                (df_mask[param_none_1] == h1) & (df_mask[param_none_2] == h2)
            ).values.flatten()
            for h1, h2 in zip(highlighted_1, highlighted_2)
        ]

        h1 = [df_mask[param_none_1].values[idx] for idx in _idx]
        h2 = [df_mask[param_none_2].values[idx] for idx in _idx]
        m = [df_mask[line_to_plot].values[idx] for idx in _idx]

        if grid:
            _h1 = np.array([_h for _h, _m in zip(h1, m) if _m == 1])
            _h2 = np.array([_h for _h, _m in zip(h2, m) if _m == 1])
            plt.scatter(
                x_op(_h1),
                x_op(_h2),
                facecolors="None",
                edgecolors="red",
                marker="s",
                s=round(5*pointsize),
                linewidth=1,
            )
            _h1 = np.array([_h for _h, _m in zip(h1, m) if _m == 0])
            _h2 = np.array([_h for _h, _m in zip(h2, m) if _m == 0])
            plt.scatter(
                x_op(_h1),
                x_op(_h2),
                facecolors="None",
                edgecolors="red",
                s=round(5*pointsize),
                linewidth=1,
            )
        elif regression:
            plt.scatter(
                x_op(np.array(highlighted_1)),
                x_op(np.array(highlighted_2)),
                c="red",
                marker="x",
            )
        else:
            _h1 = np.array([_h for _h, _m in zip(h1, m) if _m == 1])
            _h2 = np.array([_h for _h, _m in zip(h2, m) if _m == 1])
            plt.scatter(
                x_op(np.array(_h1)),
                x_op(np.array(_h2)),
                c="red",
                marker="x",
            )
            _h1 = np.array([_h for _h, _m in zip(h1, m) if _m == 0])
            _h2 = np.array([_h for _h, _m in zip(h2, m) if _m == 0])
            plt.scatter(
                x_op(np.array(_h1)),
                x_op(np.array(_h2)),
                facecolors="None",
                edgecolors="red",
                s=round(5*pointsize),
                linewidth=1,
                vmin=vmin,
                vmax=vmax,
            )

        if self._inputs_scales[k_none_1] == "log":
            plt.xscale("log")
        if self._inputs_scales[k_none_2] == "log":
            plt.yscale("log")

        plt.xlabel(
            f"${self._inputs_latex[k_none_1]}$ ({self._inputs_units_long[k_none_1]})".replace(
                "()", "(-)"
            ),
            labelpad=15,
            fontsize=int(fontsize * 1.2),
        )
        plt.ylabel(
            f"${self._inputs_latex[k_none_2]}$ ({self._inputs_units_long[k_none_2]})".replace(
                "()", "(-)"
            ),
            labelpad=15,
            fontsize=int(fontsize * 1.2),
        )
        str_title = ""
        for k in range(self.n_inputs):
            if k in [k_none_1, k_none_2]:
                continue
            str_title += (
                "${}={:.1e}$ {}, "
                if self._inputs_scales[k] == "log"
                else "${}={:.1f}$ {}, "
            ).format(self._inputs_latex[k], values[k], self._inputs_units[k])
        str_title = str_title.replace(" , ", ", ").removesuffix(", ")
        plt.title(str_title, pad=15, fontsize=int(fontsize * 1.2))
        if legend:
            leg = plt.legend(fontsize=fontsize, handletextpad=-2.0)
            for item in leg.legendHandles:
                item.set_visible(False)

        plt.gca().tick_params(axis="both", labelsize=int(fontsize))  #  * 1.2

        return cbar

    def plot_slice(
        self,
        line_to_plot: str,
        P: Optional[float] = None,
        radm: Optional[float] = None,
        Avmax: Optional[float] = None,
        angle: Optional[float] = None,
        n_samples: int = 100,
        grid: Optional[bool] = None,
        regression: Optional[bool] = None,
        errors: bool = False,
        highlighted: Optional[List[float]] = None,
        legend: bool = True,
        latex: bool = True,
        fontsize: int = 10,
        pointsize: int = 50,
        cmap: Optional[str] = None
    ) -> List[Colorbar]:
        """
        Only one variable among P, Avmax, radm and angle has to be null.
        """
        # Arguments checking
        self._check_args(n_samples, grid, regression, errors, legend, latex)

        if not isinstance(line_to_plot, str):
            raise TypeError(f"line_to_plot must be a str, not {type(line_to_plot)}")

        if regression and self._model is None:
            raise ValueError("regression must not be True if no model has been given")

        # Process Nones
        if grid is None and regression is None:
            if errors:
                grid = False
                regression = False
            else:
                grid = self._model is None
                regression = self._model is not None
        elif grid is None:
            if regression or errors:
                grid = False
            else:
                grid = True
        elif regression is None:
            if grid or errors:
                regression = False
            else:
                regression = True

        if isinstance(highlighted, dict):
            highlighted = [highlighted]
        if not isinstance(highlighted, list):
            raise TypeError(f"highlighted must be a list, not {type(highlighted)}")
        if any([not isinstance(el, dict) for el in highlighted]):
            raise ValueError(f"highlighted elements must be dict")
        if any([len(el) != 2 for el in highlighted]):
            raise ValueError(f"highlighted elements must be dict of length 2")

        # DataFrames
        # df = pd.concat([self._df_inputs, self._df_outputs], axis=1)
        # df_mask = pd.concat([self._df_inputs, self._df_mask], axis=1)

        ks_none = []
        for k in range(self.n_inputs):
            value = locals()[self.inputs_names[k]]
            if value is None:
                ks_none.append(k)

        if len(ks_none) != 2:
            raise ValueError("The number of None inputs is different from 2.")
        k_none_1, k_none_2 = ks_none
        param_none_1, param_none_2 = (
            self._inputs_names[k_none_1],
            self._inputs_names[k_none_2],
        )

        # Real lines profiles
        values = []
        list_params_filter = []
        list_closest_filter = []
        for param in self.inputs_names:
            value = locals()[param]
            if value is not None:
                closest = self.closest_in_grid(**{param: value})[param]
                values.append(closest)
                list_params_filter.append(param)
                list_closest_filter.append(closest)
            else:
                values.append(None)

        df = self._df.loc[
            (self._df[list_params_filter[0]] == list_closest_filter[0])
            & (self._df[list_params_filter[1]] == list_closest_filter[1]),
            self._inputs_names + [line_to_plot],
        ]
        df_mask = self._df_mask.loc[
            (self._df_mask[list_params_filter[0]] == list_closest_filter[0])
            & (self._df_mask[list_params_filter[1]] == list_closest_filter[1]),
            self._inputs_names + [line_to_plot],
        ]

        if errors:
            X_grid = np.zeros((len(df), self.n_inputs))
        if regression:
            X = np.zeros((n_samples**2, self.n_inputs))

        for k in range(self.n_inputs):
            value = values[k]
            if value is None and regression:
                if self._inputs_scales[k] == "log":
                    xk = np.logspace(
                        np.log10(self.grid[self.inputs_names[k]][0]),
                        np.log10(self.grid[self.inputs_names[k]][-1]),
                        n_samples,
                    )
                else:
                    xk = np.linspace(
                        self.grid[self.inputs_names[k]][0],
                        self.grid[self.inputs_names[k]][-1],
                        n_samples,
                    )
                if k == k_none_1:
                    X[:, k] = np.meshgrid(xk, xk, indexing="ij")[0].flatten()
                else:
                    X[:, k] = np.meshgrid(xk, xk, indexing="ij")[1].flatten()
            if value is None and errors:
                X_grid[:, k] = df[self.inputs_names[k]].values
            if value is not None and regression:
                X[:, k] = value
            if value is not None and errors:
                X_grid[:, k] = value

        # Neural network approximation
        if regression or errors:
            # Evaluate model
            previous_output_subset = self._model.current_output_subset
            self._model.eval()
            self._model.restrict_to_output_subset(
                [line_to_plot]
            )  # Restrict only to line we want
            y = (
                self._model.evaluate(X, transform_inputs=True, transform_outputs=True)
                if regression
                else None
            )
            y_grid = (
                self._model.evaluate(
                    X_grid, transform_inputs=True, transform_outputs=True
                )
                if regression
                else None
            )
            self._model.restrict_to_output_subset(
                previous_output_subset
            )  # Restore the previous restriction

        # Highlighted values
        highlighted_1 = [
            self.closest_in_grid(**{param_none_1: h[param_none_1]})[param_none_1]
            for h in highlighted
        ]
        highlighted_2 = [
            self.closest_in_grid(**{param_none_2: h[param_none_2]})[param_none_2]
            for h in highlighted
        ]

        # Plot slices
        kwargs = {
            "line_to_plot": line_to_plot,
            "X": X if regression else None,
            "y": y if regression else None,
            "y_grid": y_grid if errors else None,
            "values": values,
            "df": df,
            "df_mask": df_mask,
            "k_none_1": k_none_1,
            "k_none_2": k_none_2,
            "highlighted_1": highlighted_1,
            "highlighted_2": highlighted_2,
            "legend": legend,
            "fontsize": fontsize,
            "pointsize": pointsize,
            "cmap": cmap,
        }
        d = {"grid": grid, "regression": regression, "errors": errors}
        to_plot = [{param: d[param]} for param in d if d[param]]
        to_not_plot = [
            {param: False for param in d if param not in _d} for _d in to_plot
        ]

        cbars = []
        if len(to_plot) == 1:
            with LaTeX(activate=latex):
                cbar = self._plot_slice(
                    **kwargs,
                    **to_plot[0],
                    **to_not_plot[0],
                )
                cbars.append(cbar)
        elif len(to_plot) == 2:
            with LaTeX(activate=latex):
                plt.subplot(2, 1, 1)
                cbar = self._plot_slice(
                    **kwargs,
                    **to_plot[0],
                    **to_not_plot[0],
                )
                cbars.append(cbar)
                plt.tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
                plt.xlabel(None)

                plt.subplot(2, 1, 2)
                cbar = self._plot_slice(
                    **kwargs,
                    **to_plot[1],
                    **to_not_plot[1],
                )
                cbars.append(cbar)
                plt.title(None)

        elif len(to_plot) == 3:
            with LaTeX(activate=latex):
                plt.subplot(3, 1, 1)
                cbar = self._plot_slice(
                    **kwargs,
                    **to_plot[0],
                    **to_not_plot[0],
                )
                cbars.append(cbar)
                plt.tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
                plt.xlabel(None)

                plt.subplot(3, 1, 2)
                cbar = self._plot_slice(
                    **kwargs,
                    **to_plot[1],
                    **to_not_plot[1],
                )
                cbars.append(cbar)
                plt.tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
                plt.xlabel(None)
                plt.title(None)

                plt.subplot(3, 1, 3)
                cbar = self._plot_slice(
                    **kwargs,
                    **to_plot[2],
                    **to_not_plot[2],
                )
                cbars.append(cbar)
                plt.title(None)

        plt.tight_layout()

        return cbars

    def save_slices_from_csv(
        self,
        csv_file: Union[str, pd.DataFrame],
        path_outputs: str,
        n_samples: int = 100,
        grid: Optional[bool] = None,
        regression: Optional[bool] = None,
        errors: bool = False,
        legend: bool = True,
        latex: bool = True,
        dpi: int = 150,
    ) -> None:
        """TODO"""
        # Parse CSV or DataFrame
        df = self._parse_csv(csv_file)

        # Create directory
        if not os.path.isdir(path_outputs):
            os.mkdir(path_outputs)

        # Process each row
        for i in df.index:
            dirname = os.path.join(path_outputs, str(i))
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            row = df.loc[i]
            # Ignore a row if it has no line to plot
            if row.isnull()["lines"]:
                continue
            else:
                lines = [subs for subs in row["lines"].split(" ") if len(subs) > 0]
            # Ignore a row if more than two parameter are blank
            if (row[: len(self.inputs_names)].isnull()).sum() > 2:
                continue
            # If a row has exactly two blank value, we plot this profile
            if (row[: len(self.inputs_names)].isnull()).sum() == 2:
                names_blank = [row.index[row.isnull()]]
            # If a row has no blank value, we plot all the possible profiles
            else:
                names_blank = self.inputs_names
            # Save all programmed slices
            for n_blank_1, n_blank_2 in combinations(names_blank):
                d = {
                    name: (row[name] if name not in (n_blank_1, n_blank_2) else None)
                    for name in self.inputs_names
                }

                fig = plt.figure(dpi=dpi)
                self.plot_slice(
                    lines_to_plot=lines,
                    **d,
                    n_samples=n_samples,
                    grid=grid,
                    regression=regression,
                    errors=errors,
                    highlighted=[{n_blank_1: row[n_blank_1], n_blank_2: row[n_blank_2]}]
                    if len(names_blank) > 1
                    else [],
                    legend=legend,
                    latex=latex,
                )
                fig.savefig(os.path.join(dirname, f"{n_blank_1}_{n_blank_2}"))
                plt.close(fig)
