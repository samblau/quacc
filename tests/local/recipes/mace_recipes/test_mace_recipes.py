import numpy as np
import pytest
from ase.build import molecule
from maggma.stores import MemoryStore

from quacc import SETTINGS
from quacc.recipes.mace.core import freq_job, relax_job, static_job

DEFAULT_SETTINGS = SETTINGS.model_copy()


def setup_module():
    SETTINGS.PRIMARY_STORE = MemoryStore()


def teardown_module():
    SETTINGS.PRIMARY_STORE = DEFAULT_SETTINGS.PRIMARY_STORE


def test_static_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    atoms = molecule("H2O")

    output = static_job(atoms)
    assert output["natoms"] == len(atoms)
    assert output["results"]["energy"] == pytest.approx(-14.1593656539917)


def test_relax_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    atoms = molecule("H2O")

    output = relax_job(atoms)
    assert output["natoms"] == len(atoms)
    assert output["results"]["energy"] == pytest.approx(-14.160289764404297)
    assert np.max(np.linalg.norm(output["results"]["forces"], axis=1)) < 0.01

    output = relax_job(atoms, opt_params={"fmax": 0.03})
    assert output["natoms"] == len(atoms)
    assert np.max(np.linalg.norm(output["results"]["forces"], axis=1)) < 0.03


def test_freq_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    atoms = molecule("H2O")

    output = freq_job(relax_job(atoms)["atoms"])
    assert output["natoms"] == len(atoms)
    assert len(output["results"]["vib_freqs_raw"]) == 3 * len(atoms)
    assert len(output["results"]["vib_freqs"]) == 3 * len(atoms) - 6
    assert len(output["parameters_thermo"]["vib_freqs"]) == 3 * len(atoms) - 6
    assert output["parameters_thermo"]["n_imag"] == 0
