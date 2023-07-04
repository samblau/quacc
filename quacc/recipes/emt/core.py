"""
Core recipes for EMT

NOTE: This set of minimal recipes is mainly for demonstration purposes.
"""
from __future__ import annotations

import warnings

import covalent as ct
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import ExpCellFilter
from ase.optimize import FIRE

from quacc.schemas.ase import OptSchema, RunSchema, summarize_opt_run, summarize_run
from quacc.util.calc import run_ase_opt, run_calc


@ct.electron
def static_job(atoms: Atoms | dict, calc_kwargs: dict | None = None) -> RunSchema:
    """
    Carry out a static calculation.

    Parameters
    ----------
    atoms
        Atoms object or a dictionary with the key "atoms" and an Atoms object as the value
    calc_kwargs
        Dictionary of custom kwargs for the EMT calculator

    Returns
    -------
    RunSchema
        Dictionary of results from `quacc.schemas.ase.summarize_run`
    """
    atoms = atoms if isinstance(atoms, Atoms) else atoms["atoms"]
    calc_kwargs = calc_kwargs or {}

    atoms.calc = EMT(**calc_kwargs)
    final_atoms = run_calc(atoms)

    return summarize_run(
        final_atoms,
        input_atoms=atoms,
        additional_fields={"name": "EMT Static"},
    )


@ct.electron
def relax_job(
    atoms: Atoms | dict,
    relax_cell: bool = True,
    calc_kwargs: dict | None = None,
    opt_swaps: dict | None = None,
) -> OptSchema:
    """
    Carry out a geometry optimization.

    Parameters
    ----------
    atoms
        Atoms object or a dictionary with the key "atoms" and an Atoms object as the value
    relax_cell
        Whether to relax the cell
    calc_kwargs
        Dictionary of custom kwargs for the EMT calculator
    opt_swaps
        Dictionary of swaps for `run_ase_opt`
            opt_defaults = {"fmax": 0.01, "max_steps": 1000, "optimizer": FIRE}

    Returns
    -------
    OptSchema
        Dictionary of results from quacc.schemas.ase.summarize_opt_run
    """
    atoms = atoms if isinstance(atoms, Atoms) else atoms["atoms"]
    calc_kwargs = calc_kwargs or {}
    opt_swaps = opt_swaps or {}

    opt_defaults = {"fmax": 0.01, "max_steps": 1000, "optimizer": FIRE}
    opt_flags = opt_defaults | opt_swaps

    if relax_cell and not atoms.pbc.any():
        warnings.warn(
            "Volume relaxation requested but no PBCs found. Ignoring.", UserWarning
        )
        relax_cell = False

    atoms.calc = EMT(**calc_kwargs)

    if relax_cell:
        atoms = ExpCellFilter(atoms)

    dyn = run_ase_opt(atoms, **opt_flags)

    return summarize_opt_run(dyn, additional_fields={"name": "EMT Relax"})
