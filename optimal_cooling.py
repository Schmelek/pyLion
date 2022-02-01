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
ions_initial_splitting = 1e-5

# AQT Pine trap parameters

R0_eff = 0.624e-3  # in meters
Z0 = 2.25e-3  # in meters
kappa = 0.0567
RF_frequency = 25.83e6  # in Hz


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
radial_modes, radial_freqs1, axial_modes, axial_freqs1 = get_modes.get_modes(
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

norma = np.array([0.0])
order_matrix = np.zeros(len(ions_order))
set_of_orders = [
    "1000000000",
    "1100000000",
    "1110000000",
    "1111000000",
    "1111100000",
    # "111111000000",
    # "111111100000",
]
check = 0
for ions_order in set_of_orders:
    ions_order = multiset_permutations(ions_order)
    overall = ["".join(elem) for elem in ions_order]
    rev_set = set()
    tokeep = list()
    for i, elem in enumerate(overall):
        if elem not in rev_set:
            rev_set.add(elem[::-1])
            tokeep.append(i)
    ions_order = np.array(overall)[tokeep]
    ions_order = np.array([list(elem) for elem in ions_order], dtype=int)
    order_matrix = np.vstack([order_matrix, ions_order])

    for order in ions_order:
        _, radial_modes = get_modes.radial_normal_modes(
            final_z,
            DC_voltage,
            RF_voltage,
            ion_types,
            order,
            RF_frequency,
            Z0,
            R0_eff,
            kappa,
            tweezer,
            reference_ion_type,
        )
        for i in range(len(order)):
            for j in np.where(np.array(order) == 1)[0]:
                for k in np.where(np.array(order) == 0)[0]:
                    tmp = np.abs(
                        min((radial_modes[i, k]), radial_modes[i, j], key=abs)
                        / max(radial_modes[i, k], np.abs(radial_modes[i, j]), key=abs)
                    )
                    anc = np.shape(np.where(np.array(order) == 1))[1]
                    norma[-1] += tmp / (ion_number * (ion_number - anc) * anc)
                    check += 1
                    print(check)
        norma = np.append(norma, 0.0)
norma = norma[:-1]
order_matrix = order_matrix[1:, :]
order_matrix = order_matrix[np.argsort(norma), :].astype(int)
norma.sort()


plt.plot(norma, "db")
plt.grid()
plt.xlabel("Configuration index")
plt.ylabel("Norma value")
plt.title("Cooling norm for different configurations")
plt.show()

with open("matrix.txt", "w") as testfile:
    for row in order_matrix:
        testfile.write(" ".join([str(a) for a in row]) + "\n")

radial_freqs, radial_modes = get_modes.radial_normal_modes(
    final_z,
    DC_voltage,
    RF_voltage,
    ion_types,
    order_matrix[-1, :],
    RF_frequency,
    Z0,
    R0_eff,
    kappa,
    tweezer,
    reference_ion_type,
)

axial_freqs, axial_modes = get_modes.axial_normal_modes(
    final_z,
    DC_voltage,
    RF_voltage,
    ion_types,
    order_matrix[-1, :],
    RF_frequency,
    Z0,
    R0_eff,
    kappa,
    tweezer,
    reference_ion_type,
)

get_modes.comprehensive_plot(
    order_matrix[-1, :],
    data,
    radial_modes,
    axial_modes,
    radial_freqs,
    axial_freqs,
    tweezer,
)

