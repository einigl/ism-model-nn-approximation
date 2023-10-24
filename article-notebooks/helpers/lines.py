"""
Implements helpers to process spectral lines formatted according to Meudon PDR code standards.
"""

import re
from typing import List, Tuple, Union
from warnings import warn

__all__ = [
    "molecule_and_transition",
    "molecule",
    "transition",
    "is_line_of",
    "filter_molecules",
    "molecules_among_lines",
    "molecule_to_latex",
    "transition_to_latex",
    "line_to_latex",
]


## Dicts

# Molecular names to latex
_molecules_to_latex = {
    "h": "H",
    "h2": "H_2",
    "hd": "HD",
    "co": "CO",
    "13c_o": "^{13}CO",
    "c_18o": "C^{18}O",
    "13c_18o": "^{13}C^{18}O",
    "c": "C",
    "n": "N",
    "o": "O",
    "s": "S",
    "si": "Si",
    "cs": "CS",
    "cn": "CN",
    "hcn": "HCN",
    "hnc": "HNC",
    "oh": "OH",
    "h2o": "H_2O",
    "h2_18o": "H_2^{18}O",
    "c2h": "C_2H",
    "c_c3h2": "c-C_3H_2",
    "so": "SO",
    "cp": "C^+",
    "sp": "S^+",
    "hcop": "HCO^+",
    "chp": "CH^+",
    "ohp": "OH^+",
    "shp": "SH^+",
}

# Molecular names aliases
_molecules_aliases = {
    "13co": "13c_o",
    "c18o": "c_18o",
    "13c18o": "13c_18o",
    "cc3h2": "c_c3h2",
}

# Energy to LaTeX
_energy_to_latex = {
    "j": "J={}",
    "v": "\\nu={}",
    "f": "F={}",
    "n": "n={}",
    "ka": "k_a={}",
    "kc": "k_c={}",
    "fif": "F_i=F_{}",
}

# Electronic state to LaTeX
_elstate_to_latex = {
    "s": "{}s",
    "p": "{}p",
    "d": "{}d",
    "f": "{}f",
    "so": "{}S_o",
    "po": "{}P_o",
    "do": "{}D_o",
    "fo": "{}F_o",
}

# Literal to LaTeX
_literal_to_latex = {"pp": "p=+", "pm": "p=-"}


## Public functions


def molecule_and_transition(line_name: str) -> Tuple[str, str]:
    """
    Returns the raw strings of the molecule name and the transition.

    Parameters
    ----------
    line_name : str
        Formatted line.

    Returns
    -------
    str
        Raw string representing the molecule.
    str
        Raw string representing the transition.
    """
    # Check if the input is in the good format
    if not "_" in line_name:
        raise ValueError(
            f"line_name {line_name} is not in the appropriate format molecule_transition"
        )
    line_name = line_name.lower().strip()

    # Search for all matching prefixes
    prefixes = [s for s in _molecules_to_latex if line_name.startswith(s)]
    if len(prefixes) == 0:
        return tuple(line_name.split("_", maxsplit=1))

    # Select the longest prefix
    idxmax = lambda ls: max(range(len(ls)), key=ls.__getitem__)
    prefix = prefixes[idxmax([len(s) for s in prefixes])]

    # Select the remaining suffix
    suffix = line_name[len(prefix) + 1 :]

    return prefix, suffix


def molecule(line_name: str) -> str:
    """
    Returns the raw strings of the molecule name and the transition.

    Parameters
    ----------
    line_name : str
        Formatted line.

    Returns
    -------
    str
        Raw string representing the molecule.
    """
    return molecule_and_transition(line_name)[0]


def transition(line_name: str) -> str:
    """
    Returns the raw strings of the molecule name and the transition.

    Parameters
    ----------
    line_name : str
        Formatted line.

    Returns
    -------
    str
        Raw string representing the transition.
    """
    return molecule_and_transition(line_name)[1]


def molecules_among_lines(names: List[str]) -> List[str]:
    """
    Returns the list of molecules (without duplicates) of lines in `names`.

    Parameters
    ----------
    names : list of str
        List of formatted lines.

    Returns
    -------
    List of str
        List of formatted molecules without duplicates.
    """
    return list(dict.fromkeys([molecule_and_transition(name)[0] for name in names]))


def is_line_of(name: str, mol: str) -> bool:
    """
    Returns True if `name` is a line of the chemical species `mol`, else False.

    Parameters
    ----------
    name: str
        Formatted line.
    mol: str
        Molecule.

    Returns
    -------
    bool
        Whether `name` is a line of `mol`.
    """
    mol = mol.strip().lower()
    return molecule(name) == mol


def filter_molecules(names: List[str], mols: Union[str, List[str], None]) -> List[str]:
    """
    Returns a sublist of `names` with only lines of molecules contained in `mols`.

    Parameters
    ----------
    names : list of str
        List of formatted lines.
    mols : str or List of str or None
        Molecule or list of molecules that you want to select. If None, the function just returns the input list `names`.

    Returns
    -------
    List of str
        Sublist of `names`.
    """
    if mols is None:
        return names

    if isinstance(mols, str):
        mols = [mols]
    for i, mol in enumerate(mols):
        mols[i] = mol.strip().lower()

    mols = [
        (_molecules_aliases[mol] if mol in _molecules_aliases else mol) for mol in mols
    ]

    lines_mols = [molecule(name) for name in names]
    indices = [i for i, line_mol in enumerate(lines_mols) if line_mol in mols]

    return [names[i] for i in indices]


def molecule_to_latex(molecule: str) -> str:
    """
    Returns a well displayed version of the formatted molecule or radical `molecule`.

    Parameters
    ----------
    molecule : str
        Formatted molecule or radical.

    Returns
    -------
    str
        LaTeX string representing `molecule`.
    """
    if molecule in _molecules_to_latex:
        latex_molecule = "$\\mathrm{{{}}}$".format(_molecules_to_latex[molecule])
    else:
        latex_molecule = molecule

    return latex_molecule


def transition_to_latex(transition: str) -> str:
    """
    Returns a well displayed version of the formatted transition `transition`.

    Parameters
    ----------
    transition : str
        Formatted transition.

    Returns
    -------
    str
        LaTeX string representing `transition`.
    """
    if transition.count("__") != 1:
        raise ValueError(
            f"{transition} is not a valid transition because it does not contain one occurence of the double underscore"
        )
    high, low = transition.split("__")

    names = []
    high_lvls, low_lvls = [], []
    while high != "" and low != "":
        # Match energy levels
        res_high = re.match("\A(fif|j|v|n|f|ka|kc)(\d*_\d\d*|\d*d\d*|\d*)", high)
        res_low = re.match("\A(fif|j|v|n|f|ka|kc)(\d*_\d\d*|\d*d\d*|\d*)", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[: res_high.end()], low[: res_low.end()]
            n_high = re.match("\A(fif|j|v|n|f|ka|kc)", e_high).group()
            n_low = re.match("\A(fif|j|v|n|f|ka|kc)", e_low).group()
            if n_high != n_low:
                raise ValueError(
                    "{transition} is not a valid transition because the energy levels are not in the same order in the description of the high and low levels"
                )
            names.append(n_high)
            high_lvls.append(
                _removeprefixes(e_high, n_high).replace("_", "/").replace("d", ".")
            )
            low_lvls.append(
                _removeprefixes(e_low, n_low).replace("_", "/").replace("d", ".")
            )
            high = _removeprefixes(high, e_high, "_")
            low = _removeprefixes(low, e_low, "_")
            continue

        # Match electronic state
        res_high = re.match("\Ael\d*(po|so|do|p|s|d)", high)
        res_low = re.match("\Ael\d*(po|so|do|p|s|d)", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[: res_high.end()], low[: res_low.end()]
            names.append("el")
            high_lvls.append(_removeprefixes(e_high, "el"))
            low_lvls.append(_removeprefixes(e_low, "el"))
            high = _removeprefixes(high, e_high, "_")
            low = _removeprefixes(low, e_low, "_")
            continue

        # Match literals
        res_high = re.match("\A(pp|pm)", high)
        res_low = re.match("\A(pp|pm)", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[: res_high.end()], low[: res_low.end()]
            names.append("lit")
            high_lvls.append(e_high)
            low_lvls.append(e_low)
            high = _removeprefixes(high, e_high, "_")
            low = _removeprefixes(low, e_low, "_")
            continue

        if high == "" and low != "" or high != "" and low == "":
            raise RuntimeError(
                "high and low levels does not contain the same number of variables"
            )

    return _sort_transitions(names, high_lvls, low_lvls)


def line_to_latex(line_name: str) -> str:
    """
    Returns a well displayed version of the formatted line `line_name`.

    Parameters
    ----------
    line_name : str
        Formatted line.

    Returns
    -------
    str
        LaTeX string representing `line_name`.
    """

    prefix, suffix = molecule_and_transition(line_name)

    # Convert the prefix in LaTeX
    latex_prefix = molecule_to_latex(prefix)

    # Convert the suffix in LaTeX
    latex_suffix = transition_to_latex(suffix)

    out = latex_prefix + " " + latex_suffix
    out = out.replace("  ", " ")  # Remove double spaces
    return out


# Local functions


def _removeprefixes(string: str, *prefixes: str) -> str:
    """
    Return a str with the given prefix string removed if present.

    Return a copy of `string` with the prefixes `prefixes` removed iteratively if they exists.

    Note
    ----

    This method doesn't use the builtin method `removeprefix` to ensure the code to be available to users with `Python < 3.9`.
    """
    for prefix in prefixes:
        if string.startswith(prefix):
            string = string[len(prefix) :]
    return string[:]


def _numerical_to_latex(num: str) -> str:
    """
    Returns a LaTeX string representing a numerical value `num`. This value can be formatted in several ways: `'a'`, `'a/b'` or `'a.b'` where a and b are integers, potentially over several digits.

    Parameters
    ----------
    num : str
        Formatted number.

    Returns
    -------
    str
        LaTeX representation of the number.
    """

    if re.match("\A\d*[/.]\d*\Z", num) is None:
        return num

    if "/" in num:
        a, b = num.split("/")
        n, d = int(a), int(b)
    else:
        a, b = num.split(".")

        if b == "0":
            n, d = int(a), 1
        elif b == "5":
            n, d = 2 * int(a) + 1, 2
        else:
            warn(f"x.{b} floats has not been implemented. Ignoring the floating part.")
            n, d = int(a), 1  # Default behavior

    if n % d == 0:
        num_latex = f"{n // d}"
    else:
        num_latex = r"\frac{" + str(n) + r"}{" + str(d) + r"}"

    return num_latex


def _transition(name: str, high_lvl: str, low_lvl: str) -> Tuple[str, str]:
    """
    Returns a LaTeX string representing a non electronic transition.

    Parameters
    ----------
    name : str
        Energy name.
    high_lvl : str
        Higher energy level.
    low_lvl : str
        Lower energy level. Can be the same as `high_lvl`.

    Returns
    -------
    str
        Higher energy level formatted in LaTeX.
    str
        Lower energy level formatted in LaTeX. May be the same as the higher level.
    """
    if name in _energy_to_latex:
        name_latex = _energy_to_latex[name]
    else:
        name_latex = name + "={}"  # Default behavior for unknown name

    high_lvl_latex = _numerical_to_latex(high_lvl)
    low_lvl_latex = _numerical_to_latex(low_lvl)

    return (
        "$" + name_latex.format(high_lvl_latex) + "$",
        "$" + name_latex.format(low_lvl_latex) + "$",
    )


def _eltransition(high: str, low: str) -> Tuple[str, str]:
    """
    Returns a LaTeX string representing an electronic transition.

    Parameters
    ----------
    high : str
        Higher energy electronic configuration.
    low : str
        Lower energy electronic configuration. Can be the same as `high`.

    Returns
    -------
    str
        Higher energy electronic configuration formatted in LaTeX.
    str
        Lower energy electronic configuration formatted in LaTeX. May be the same as the higher configuration.
    """
    num_high, orb_high = high[0], high[1:]
    num_low, orb_low = low[0], low[1:]
    return (
        "$"
        + (
            _elstate_to_latex[orb_high].format(num_high)
            if orb_high in _elstate_to_latex
            else high
        )
        + "$",
        "$"
        + (
            _elstate_to_latex[orb_low].format(num_low)
            if orb_low in _elstate_to_latex
            else low
        )
        + "$",
    )


def _littransition(high: str, low: str) -> Tuple[str, str]:
    """
    Returns a LaTeX string representing whatever transition.

    Parameters
    ----------
    high : str
        Higher configuration.
    low : str
        Lower configuration. Can be the same as `high`.

    Returns
    -------
    str
        Higher configuration formatted in LaTeX.
    str
        Lower configuration formatted in LaTeX.
    """
    return (
        "${}$".format(_literal_to_latex[high] if high in _literal_to_latex else high),
        "${}$".format(_literal_to_latex[low] if low in _literal_to_latex else low),
    )


def _sort_transitions(
    names: List[str], high_lvls: List[int], low_lvls: List[int]
) -> str:
    """
    Returns a LaTeX string representing the energy transitions.
    This string first display the constant energy levels and then the energy transitions.

    Parameters
    ----------
    names : list of str
        Energies names.
    high_lvls : list of int
        List of higher level for each energy.
    low_lvls : List of int.
        List of lower level for each energy.

    Returns
    -------
    str
        String representing first the constant energy levels and then the energy transitions.
    """
    if len(high_lvls) != len(names) or len(low_lvls) != len(names):
        raise ValueError("names, high_lvls and low_lvls must have the same length")

    if len(names) == 0:
        return ""

    descr_a, descr_b = "", ""
    for name, high, low in zip(names, high_lvls, low_lvls):
        if name == "lit":
            descr = _littransition(high, low)
        elif name == "el":
            descr = _eltransition(high, low)
        else:
            descr = _transition(name, high, low)
        descr_a += descr[0] + ", "
        descr_b += descr[1] + ", "

    return "({} $\\to$ {})".format(descr_a[:-2], descr_b[:-2])
