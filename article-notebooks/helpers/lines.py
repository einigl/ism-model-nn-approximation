import re
from typing import List, Tuple, Union
from warnings import warn

__all__ = [
    "molecule_and_transition",
    "molecule",
    "transition",
    "filter_molecules",
    "molecules_among_lines",
    "molecule_to_latex",
    "transition_to_latex",
    "line_to_latex"
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
    "j": "J",
    "v": "\\nu",
    "f": "f",
    "n": "n",
    "ka": "k_a",
    "kc": "k_c",
}


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
        raise ValueError(f'line_name {line_name} is not in the appropriate format molecule_transition')
    line_name = line_name.lower().strip()

    # Search for all matching prefixes
    prefixes = [s for s in _molecules_to_latex if line_name.startswith(s)]
    if len(prefixes) == 0:
        return tuple(line_name.split('_', maxsplit=1))

    # Select the longest prefix
    idxmax = lambda ls: max(range(len(ls)), key=ls.__getitem__)
    prefix = prefixes[idxmax([len(s) for s in prefixes])]

    # Select the remaining suffix
    suffix = line_name[len(prefix)+1:]

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

def filter_molecules(
    names: List[str], mols: Union[str, List[str], None]
) -> List[str]:
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

    mols = [(_molecules_aliases[mol] if mol in _molecules_aliases else mol) for mol in mols]

    lines_mols = [molecule(name) for name in names]
    indices = [i for i, line_mol in enumerate(lines_mols) if line_mol in mols]

    return [names[i] for i in indices]

def molecule_to_latex(molecule: str) -> str:
    """
    Returns a well displayed version of the formatted molecule or radical `molecule`.

    Not addressed formats :
    * '(pp|pm)_fif\d*'

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
        latex_molecule = "${}$".format(_molecules_to_latex[molecule])
    else:
        latex_molecule = molecule

    return latex_molecule

def transition_to_latex(transition: str) -> str:
    """
    Returns a well displayed version of the formatted transition `transition`.

    Not addressed formats :
    * '(pp|pm)_fif\d*'

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
        raise ValueError(f"{transition} is not a valid transition because it does not contain one occurence of the double underscore")
    high, low = transition.split("__")

    names = []
    high_lvls, low_lvls = [], []
    while high != "" and low != "":

        res_high = re.match("\A(j|v|n|f|ka|kc)\d*_2", high)
        res_low = re.match("\A(j|v|n|f|ka|kc)\d*_2", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[:res_high.end()], low[:res_low.end()]
            n_high = re.match("\A(j|v|n|f|ka|kc)", e_high).group()
            n_low = re.match("\A(j|v|n|f|ka|kc)", e_low).group()
            if n_high != n_low:
                raise ValueError("{transition} is not a valid transition because the energy level are not in the same order in the description of the high and low levels")
            names.append(n_high)
            high_lvls.append(e_high.removeprefix(n_high).replace('_', '/'))
            low_lvls.append(e_low.removeprefix(n_low).replace('_', '/'))
            high = high.removeprefix(e_high).removeprefix('_')
            low = low.removeprefix(e_low).removeprefix('_')
            continue

        res_high = re.match("\A(j|v|n|f|ka|kc)\d*d\d*", high)
        res_low = re.match("\A(j|v|n|f|ka|kc)\d*d\d*", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[:res_high.end()], low[:res_low.end()]
            n_high = re.match("\A(j|v|n|f|ka|kc)", e_high).group()
            n_low = re.match("\A(j|v|n|f|ka|kc)", e_low).group()
            if n_high != n_low:
                raise ValueError("{transition} is not a valid transition because the energy level are not in the same order in the description of the high and low levels")
            names.append(n_high)
            high_lvls.append(e_high.removeprefix(n_high).replace('d', '.'))
            low_lvls.append(e_low.removeprefix(n_low).replace('d', '.'))
            high = high.removeprefix(e_high).removeprefix('_')
            low = low.removeprefix(e_low).removeprefix('_')
            continue

        res_high = re.match("\A(j|v|n|f|ka|kc)\d*", high)
        res_low = re.match("\A(j|v|n|f|ka|kc)\d*", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[:res_high.end()], low[:res_low.end()]
            n_high = re.match("\A(j|v|n|f|ka|kc)", e_high).group()
            n_low = re.match("\A(j|v|n|f|ka|kc)", e_low).group()
            if n_high != n_low:
                raise ValueError("{transition} is not a valid transition because the energy level are not in the same order in the description of the high and low levels")
            names.append(n_high)
            high_lvls.append(e_high.removeprefix(n_high))
            low_lvls.append(e_low.removeprefix(n_low))
            high = high.removeprefix(e_high).removeprefix('_')
            low = low.removeprefix(e_low).removeprefix('_')
            continue

        res_high = re.match("\Ael\d*(po|so|do)", high)
        res_low = re.match("\Ael\d*(po|so|do)", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[:res_high.end()], low[:res_low.end()]
            names.append("el")
            high_lvls.append(e_high.removeprefix("el"))
            low_lvls.append(e_low.removeprefix("el"))
            high = high.removeprefix(e_high).removeprefix('_')
            low = low.removeprefix(e_low).removeprefix('_')
            continue

        res_high = re.match("\Ael\d*(p|s|d)", high)
        res_low = re.match("\Ael\d*(p|s|d)", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[:res_high.end()], low[:res_low.end()]
            names.append("el")
            high_lvls.append(e_high.removeprefix("el"))
            low_lvls.append(e_low.removeprefix("el"))
            high = high.removeprefix(e_high).removeprefix('_')
            low = low.removeprefix(e_low).removeprefix('_')
            continue

        res_high = re.match("\A(pp|pm)_fif\d*", high)
        res_low = re.match("\A(pp|pm)_fif\d*", low)
        if res_high is not None and res_low is not None:
            e_high, e_low = high[:res_high.end()], low[:res_low.end()]
            # names.append("el")
            # high_lvls.append(e_high.removeprefix("el"))
            # low_lvls.append(e_low.removeprefix("el"))
            high = high.removeprefix(e_high).removeprefix('_')
            low = low.removeprefix(e_low).removeprefix('_')
            continue

        if high == "" and low != "" or high != "" and low == "":
            raise RuntimeError("high and low levels does not contain the same number of variables")

    return _sort_transitions(names, high_lvls, low_lvls)

def line_to_latex(line_name: str) -> str:
    """
    Returns a well displayed version of the formatted line `line_name`.

    Not addressed formats :
    * '(pp|pm)_fif\d*'

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
    out = out.replace("  ", " ") # Remove double spaces
    return out


# Local functions

def _transition(
    name: str, high_lvl: str, low_lvl: str
) -> Tuple[Union[str, Tuple[str, str]], bool]:
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
    str or tuple of str
        If it is a transition: higher energy level and lower energy level. Else: energy level.
    bool
        True if it is a transition, else False
    """
    if name in _energy_to_latex:
        name_latex = _energy_to_latex[name]
    else:
        name_latex = name

    if re.match("\A\d*[/.]\d*\Z", high_lvl) is not None:
        if '/' in high_lvl:
            a_high, b_high = high_lvl.split('/')
            a_low, b_low = low_lvl.split('/')

            n_high, d_high = int(a_high), int(b_high)
            n_low, d_low = int(a_low), int(b_low)
        else:
            a_high, b_high = high_lvl.split('.')
            a_low, b_low = low_lvl.split('.')

            if b_high == "0": 
                n_high, d_high = 2*int(a_high), 1
            elif b_high == "5":
                n_high, d_high = 2*int(a_high)+1, 2
            else:
                warn(f"x.{b_high} floats has not been implemented. Ignoring the floating part.")
                n_high, d_high = int(a_high), 1 # Default behavior
            if b_low == "0": 
                n_low, d_low = 2*int(a_low), 1
            elif b_low == "5":
                n_low, d_low = 2*int(a_low)+1, 2
            else:
                warn(f"x.{b_low} floats has not been implemented. Ignoring the floating part.")
                n_low, d_low = int(a_low), 1 # Default behavior

        if n_high % d_high == 0:
            high_lvl_latex = f"{n_high // d_high}"
        else:
            high_lvl_latex = r"\frac{" + str(n_high) + r"}{" + str(d_high) + r"}"
        if n_low % d_low == 0:
            low_lvl_latex = f"{n_low // d_low}"
        else:
            low_lvl_latex = r"\frac{" + str(n_low) + r"}{" + str(d_low) + r"}"
    else:
        high_lvl_latex = high_lvl
        low_lvl_latex = low_lvl

    if high_lvl_latex == low_lvl_latex:
        return "${}={}$".format(name_latex, low_lvl_latex), False
    return (
        "${}={}$".format(name_latex, high_lvl_latex),
        "${}={}$".format(name_latex, low_lvl_latex),
    ), True

def _eltransition(high: str, low: str) -> Tuple[Union[str, Tuple[str, str]], bool]:
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
    str or tuple of str
        If it is a transition: higher energy electronic configuration and lower energy configuration. Else: energy electronic configuration.
    bool
        True if it is a transition, else False
    """
    if high == low:
        return "${}$".format(high), False
    return ("${}$".format(high), "${}$".format(low)), True


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
    # if len(names) == 1:
    #     return _transition(None, high_lvls[0], low_lvls[0])

    descr_0, descr_1a, descr_1b = "", "", ""
    for name, high, low in zip(names, high_lvls, low_lvls):
        if name == "el":
            descr, istrans = _eltransition(high, low)
        else:
            descr, istrans = _transition(name, high, low)
        if istrans:
            descr_1a += descr[0] + ", "
            descr_1b += descr[1] + ", "
        else:
            descr_0 += descr + " "
    return "{} ({} $\\to$ {})"\
        .format(descr_0.strip(), descr_1a[:-2], descr_1b[:-2])

# def _transition_to_latex(transition: str) -> str:
#     """
#     TODO
#     """
#     if re.match("\Aj\d*__j\d*\Z", transition):
#         # ??__j{}__j{}
#         res = re.search("\Aj(.*?)__j(.*?)\Z", transition)
#         if res:
#             # latex_transition = "$({} \\to {})$".format(*res.group(1, 2))
#             latex_transition = "$(J={} \\to J={})$".format(*res.group(1, 2))
#         else:
#             raise ValueError("Should never been here")
#     elif re.match("\Av\d*_j\d*__v\d*_j\d*\Z", transition):
#         # ??_v{}_j{}__v{}_j{}
#         res = re.search("\Av(.*?)_j(.*?)__v(.*?)_j(.*?)\Z", transition)
#         if res:
#             names = ["v", "j"]
#             high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
#             latex_transition = _sort_transitions(names, high_lvls, low_lvls)
#         else:
#             raise ValueError("Should never been here")
#     elif re.match("\An\d*_j\d*__n\d*_j\d*\Z", transition):
#         # ??_n{}_j{}__n{}_j{}
#         res = re.search("\An(.*?)_j(.*?)__n(.*?)_j(.*?)\Z", transition)
#         if res:
#             names = ["n", "j"]
#             high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
#             latex_transition = _sort_transitions(names, high_lvls, low_lvls)
#         else:
#             raise ValueError("Should never been here")
#     elif re.match("\Aj\d*_f\d*__j\d*_f\d*\Z", transition):
#         # ??_j{}_f{}__j{}_f{}
#         res = re.search("\Aj(.*?)_f(.*?)__j(.*?)_f(.*?)\Z", transition)
#         if res:
#             names = ["j", "f"]
#             high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
#             latex_transition = _sort_transitions(names, high_lvls, low_lvls)
#         else:
#             raise ValueError("Should never been here")
#     elif re.match("\Aj\d*_ka\d*_kc\d*__j\d*_ka\d*_kc\d*\Z", transition):
#         # ??_j{}_ka{}_kc{}__j{}_ka{}_kc{}
#         res = re.search("\Aj(.*?)_ka(.*?)_kc(.*?)__j(.*?)_ka(.*?)_kc(.*?)\Z", transition)
#         if res:
#             names = ["j", "ka", "kc"]
#             high_lvls, low_lvls = res.group(1, 2, 3), res.group(4, 5, 6)
#             latex_transition = _sort_transitions(names, high_lvls, low_lvls)
#         else:
#             raise ValueError("Should never been here")
#     elif re.match("\Ael\d*\w_j\d*__el\d*\w_j\d*\Z", transition):
#         # ??_el{int+char}_j{int}__el{int+char}_j{int}
#         res = re.search("\Ael(.*?)_j(.*?)__el(.*?)_j(.*?)\Z", transition)
#         if res:
#             names = ["el", "j"]
#             high_lvls, low_lvls = res.group(1, 2), res.group(3, 4)
#             latex_transition = _sort_transitions(names, high_lvls, low_lvls)
#     elif re.match("\Ael\d*\w_j\d*_2__el\d*\w_j\d*_2\Z", transition):
#         # ??_el{int+char}_j{int}_2__el{int+char}_j{int}_2
#         res = re.search("\Ael(.*?)_j(.*?)_2__el(.*?)_j(.*?)_2\Z", transition)
#         if res:
#             names = ["el", "j"]
#             high_lvls = res.group(1), r"\frac{" + f"{res.group(2)}" + r"}{2}"
#             low_lvls = res.group(3), r"\frac{" + f"{res.group(4)}" + r"}{2}"
#             latex_transition = _sort_transitions(names, high_lvls, low_lvls)
#     else:
#         # Not adressed formats

#         # oh_j5_2_pp_fif2__j3_2_pm_fif2

#         # c2h_n1d0_j1d5_f1d0__n0d0_j0d5_f1d0

#         # cn_n1_j0d5__n0_j0d5
#         # ohp_n1_j0_f0d5__n0_j1_f0d5
#         # shp_n1_j0_f0d5__n0_j1_f0d5

#         latex_transition = transition

#     return latex_transition
