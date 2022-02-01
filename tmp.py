from webbrowser import get
from jinja2.sandbox import F
from numpy.core.fromnumeric import argmax
import pylion as pl
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import get_modes
from sympy.utilities.iterables import multiset_permutations


# Constants declaration
ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity


# use filename for simulation name
# name = Path(__file__).stem


# Declaration of ion types used in the simulation
ion_types = [{"mass": 40, "charge": 1}, {"mass": 43, "charge": 1}]
# Ions ordering. You should place ion type number from the previous list in a desired order.
ions_order = [0] * 10
tw = 1e6
tweezer = [0] * 10
ions_order = np.array(ions_order)
ion_number = ions_order.shape[0]
# Initial distance between two neighboring ions
ions_initial_splitting = 1e-4


# AQT Pine trap parameters

R0_eff = 0.624e-3  # in meters
Z0 = 2.25e-3  # in meters
kappa = 0.0567
RF_frequency = 25e6  # in Hz


reference_ion_type = 0  # number of ion's type for each frequencies are defined

# If you want to operate with secular frequencies, but not voltages, uncomment block bellow
# w_z = 0.2e6  # axial secular frequency in Hz
# w_r = 2e6  # radial secular frequency in Hz

# DC_voltage, RF_voltage = get_modes.frequency_to_voltage(
#     w_z, w_r, ion_types[reference_ion_type], RF_frequency, Z0, R0_eff, kappa
# )
# print("Endcap voltage:", DC_voltage, "V, blade voltage:", RF_voltage, "V")

DC_voltage = 3
RF_voltage = 850


# Description of Paul trap parameters
trap = {
    "radius": R0_eff,
    "length": Z0,
    "kappa": kappa,
    "frequency": RF_frequency,
    "voltage": RF_voltage,
    "endcapvoltage": DC_voltage,
    "pseudo": True,
    "anisotropy": 1,
}

"""Simulation of ion crystal structure"""


# for current_order in ions_order:
radial_modes, radial_freqs, axial_modes, axial_freqs = get_modes.get_modes(
    ion_types,
    ions_order,
    ion_number,
    ions_initial_splitting,
    trap,
    tweezer,
    reference_ion_type,
)
_, data = pl.readdump("positions.txt")
final_z = data[-1, :, 5]
final_z = np.sort(final_z)

_, radial_modes = get_modes.radial_normal_modes(
    final_z,
    DC_voltage,
    RF_voltage,
    ion_types,
    ions_order,
    RF_frequency,
    Z0,
    R0_eff,
    kappa,
    tweezer,
    reference_ion_type,
)

