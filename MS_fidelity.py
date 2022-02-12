from dis import disco
from webbrowser import get
from jinja2.sandbox import F
from numpy.core.fromnumeric import argmax
import pylion as pl
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import get_modes
from scipy.optimize import fsolve
from numpy import complex128, sin as sin
from numpy import cos as cos
import MS_em

# Constants declaration
ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity
c = 299792458
hbar = 1.054571817e-34

# use filename for simulation name
# name = Path(__file__).stem


# Declaration of ion types used in the simulation
ion_types = [{"mass": 171, "charge": 1}]
# Ions ordering. You should place ion type number from the previous list in a desired order.
ions_order = [0] * 5
tw = 1e6
tweezer = [0] * 5
ions_order = np.array(ions_order)
ion_number = ions_order.shape[0]
# Initial distance between two neighboring ions
ions_initial_splitting = 1e-5


# AQT Pine trap parameters

R0_eff = 0.624e-3  # in meters
Z0 = 2.25e-3  # in meters
kappa = 0.0567
RF_frequency = 20.83e6  # in Hz


reference_ion_type = 0  # number of ion's type for each frequencies are defined

# If you want to operate with secular frequencies, but not voltages, uncomment block bellow
w_z = 2.7e5  # axial secular frequency in Hz
w_r = 3.0692e6  # radial secular frequency in Hz

DC_voltage, RF_voltage = get_modes.frequency_to_voltage(
    w_z, w_r, ion_types[reference_ion_type], RF_frequency, Z0, R0_eff, kappa
)
print("Endcap voltage:", DC_voltage, "V, blade voltage:", RF_voltage, "V")

# DC_voltage = 3
# RF_voltage = 850


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
l = ((1 * ech) ** 2 / (171 * amu * 4 * np.pi * eps0 * (w_z * 2 * np.pi) ** 2)) ** (
    1 / 3
)

radial_freqs = radial_freqs[::-1] * w_z * 2 * np.pi
radial_modes = -radial_modes[::-1, :]
radial_freqs = np.array([3.0692, 3.0522, 3.0344, 3.0127, 2.9867])*1e6*2*np.pi


S = 2 * ion_number + 1
offset = 3.044e6*2*np.pi

tau = 235e-6
C1 = np.zeros((ion_number, S), dtype=np.complex128)
C2 = np.zeros((ion_number, S), dtype=np.complex128)

Dij = np.zeros((S, S))
LD_parameter = np.zeros((ion_number, 2))
ions = [0, 1]  # mind that in python numbers begin with 0!
for i in range(ion_number):
    for j in range(2):
        LD_parameter[i, j] = (
            2
            * 2
            * np.pi
            / 355e-9
            * (
                hbar
                / (2 * ion_types[ions_order[ions[j]]]["mass"] * amu * radial_freqs[i])
            )
            ** 0.5
        ) * radial_modes[i, ions[j]]

for i in range(ion_number):
    for j in range(S):
        t1 = j * tau / S
        t2 = (j + 1) * tau / S
        w = radial_freqs[i]
        C1[i, j] = (
            -offset * cos(offset * t1) * cos(t1 * w) /
            (-(offset ** 2) + w ** 2)
            + offset * cos(offset * t2) * cos(t2 * w) /
            (-(offset ** 2) + w ** 2)
            - w * sin(offset * t1) * sin(t1 * w) / (-(offset ** 2) + w ** 2)
            + w * sin(offset * t2) * sin(t2 * w) / (-(offset ** 2) + w ** 2)
            + 1j
            * (
                -offset * sin(t1 * w) * cos(offset * t1) /
                (-(offset ** 2) + w ** 2)
                + offset * sin(t2 * w) * cos(offset * t2) /
                (-(offset ** 2) + w ** 2)
                + w * sin(offset * t1) * cos(t1 * w) /
                (-(offset ** 2) + w ** 2)
                - w * sin(offset * t2) * cos(t2 * w) /
                (-(offset ** 2) + w ** 2)
            )
        ) * LD_parameter[i, 0]

# for i in range(ion_number):
#     for j in range(S):
#         tmp1 = integrate.quad(
#             lambda x: np.sin(offset * x) * np.cos(radial_freqs[i] * x),
#             j * tau / S,
#             (j + 1) * tau / S,
#         )[0]
#         tmp2 = integrate.quad(
#             lambda x: np.sin(offset * x) * np.sin(radial_freqs[i] * x),
#             j * tau / S,
#             (j + 1) * tau / S,
#         )[0]
#         C2[i, j] = (tmp1 + 1j * tmp2) * LD_parameter[i, 1]


for n in range(S):
    for m in range(n):
        t1 = n * tau / S
        t2 = (n + 1) * tau / S
        t1p = m * tau / S
        t2p = (m + 1) * tau / S

        def tmp(w): return (
            -(offset ** 2)
            * sin(t1 * w)
            * cos(offset * t1)
            * cos(offset * t1p)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t1 * w)
            * cos(offset * t1)
            * cos(offset * t2p)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t1p * w)
            * cos(offset * t1)
            * cos(offset * t1p)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset ** 2
            * sin(t1p * w)
            * cos(offset * t1p)
            * cos(offset * t2)
            * cos(t2 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t2 * w)
            * cos(offset * t1p)
            * cos(offset * t2)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset ** 2
            * sin(t2 * w)
            * cos(offset * t2)
            * cos(offset * t2p)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset ** 2
            * sin(t2p * w)
            * cos(offset * t1)
            * cos(offset * t2p)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset ** 2
            * sin(t2p * w)
            * cos(offset * t2)
            * cos(offset * t2p)
            * cos(t2 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1)
            * sin(t1 * w)
            * sin(t1p * w)
            * cos(offset * t1p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1)
            * sin(t1 * w)
            * sin(t2p * w)
            * cos(offset * t2p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1)
            * cos(offset * t1p)
            * cos(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1)
            * cos(offset * t2p)
            * cos(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1p)
            * sin(t1 * w)
            * sin(t1p * w)
            * cos(offset * t1)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1p)
            * sin(t1p * w)
            * sin(t2 * w)
            * cos(offset * t2)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t1p)
            * cos(offset * t1)
            * cos(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t1p)
            * cos(offset * t2)
            * cos(t1p * w)
            * cos(t2 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t2)
            * sin(t1p * w)
            * sin(t2 * w)
            * cos(offset * t1p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2)
            * sin(t2 * w)
            * sin(t2p * w)
            * cos(offset * t2p)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t2)
            * cos(offset * t1p)
            * cos(t1p * w)
            * cos(t2 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2)
            * cos(offset * t2p)
            * cos(t2 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2p)
            * sin(t1 * w)
            * sin(t2p * w)
            * cos(offset * t1)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t2p)
            * sin(t2 * w)
            * sin(t2p * w)
            * cos(offset * t2)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + offset
            * w
            * sin(offset * t2p)
            * cos(offset * t1)
            * cos(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - offset
            * w
            * sin(offset * t2p)
            * cos(offset * t2)
            * cos(t2 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t1)
            * sin(offset * t1p)
            * sin(t1 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t1)
            * sin(offset * t1p)
            * sin(t1p * w)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t1)
            * sin(offset * t2p)
            * sin(t1 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t1)
            * sin(offset * t2p)
            * sin(t2p * w)
            * cos(t1 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t1p)
            * sin(offset * t2)
            * sin(t1p * w)
            * cos(t2 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t1p)
            * sin(offset * t2)
            * sin(t2 * w)
            * cos(t1p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            - w ** 2
            * sin(offset * t2)
            * sin(offset * t2p)
            * sin(t2 * w)
            * cos(t2p * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
            + w ** 2
            * sin(offset * t2)
            * sin(offset * t2p)
            * sin(t2p * w)
            * cos(t2 * w)
            / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        )
        Dij[n, m] = sum(
            tmp(radial_freqs[a]) * LD_parameter[a, 0] * LD_parameter[a, 1]
            for a in range(ion_number)
        )
        Dij[m, n] = Dij[n, m]


for n in range(S):
    t1p = n * tau / S
    t2p = (n + 1) * tau / S
    t1 = n * tau / S

    def tmp(w): return (
        -(offset ** 2)
        * sin(t1 * w)
        * cos(offset * t1)
        * cos(offset * t1p)
        * cos(t1p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + offset ** 2
        * sin(t1 * w)
        * cos(offset * t1)
        * cos(offset * t2p)
        * cos(t2p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + offset ** 2
        * sin(t1p * w)
        * cos(offset * t1)
        * cos(offset * t1p)
        * cos(t1 * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        - offset ** 2
        * sin(t2p * w)
        * cos(offset * t1)
        * cos(offset * t2p)
        * cos(t1 * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + offset
        * w
        * sin(offset * t1)
        * sin(t1 * w)
        * sin(t1p * w)
        * cos(offset * t1p)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        - offset
        * w
        * sin(offset * t1)
        * sin(t1 * w)
        * sin(t2p * w)
        * cos(offset * t2p)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + offset
        * w
        * sin(offset * t1)
        * cos(offset * t1p)
        * cos(t1 * w)
        * cos(t1p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        - offset
        * w
        * sin(offset * t1)
        * cos(offset * t2p)
        * cos(t1 * w)
        * cos(t2p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        - offset
        * w
        * sin(offset * t1p)
        * sin(t1 * w)
        * sin(t1p * w)
        * cos(offset * t1)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        - offset
        * w
        * sin(offset * t1p)
        * cos(offset * t1)
        * cos(t1 * w)
        * cos(t1p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + offset
        * w
        * sin(offset * t2p)
        * sin(t1 * w)
        * sin(t2p * w)
        * cos(offset * t1)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + offset
        * w
        * sin(offset * t2p)
        * cos(offset * t1)
        * cos(t1 * w)
        * cos(t2p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        - w ** 2
        * sin(offset * t1)
        * sin(offset * t1p)
        * sin(t1 * w)
        * cos(t1p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + w ** 2
        * sin(offset * t1)
        * sin(offset * t1p)
        * sin(t1p * w)
        * cos(t1 * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        + w ** 2
        * sin(offset * t1)
        * sin(offset * t2p)
        * sin(t1 * w)
        * cos(t2p * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
        - w ** 2
        * sin(offset * t1)
        * sin(offset * t2p)
        * sin(t2p * w)
        * cos(t1 * w)
        / (offset ** 4 - 2 * offset ** 2 * w ** 2 + w ** 4)
    )
    Dij[n, n] = sum(
        LD_parameter[a, 0] * LD_parameter[a, 1] * tmp(radial_freqs[a])
        for a in range(ion_number)
    )

C_matrix = np.vstack([C1.imag, C1.real])
Dij = Dij


def equations(omega):
    omega = np.array(omega)
    out = np.concatenate(
        (np.dot(C_matrix, omega), np.dot(omega, omega) - 1), axis=None,
    )
    return out.T


omega = fsolve(equations, np.arange(S))
if np.dot(omega, (np.dot(Dij, omega))) < 0:
    omega = omega * \
        (-np.pi * 7 / (4 * np.dot(omega, (np.dot(Dij, omega))))) ** 0.5
else:
    omega = omega * (np.pi / (4 * np.dot(omega, (np.dot(Dij, omega))))) ** 0.5
print("Omega", omega)


plt.bar(np.arange(S), omega, color="b")
plt.grid()
plt.show()


def omega_function(t): return np.piecewise(
    t, [((i * tau / S <= t) & (t < (i + 1) * tau / S))
        for i in range(S)], omega
)

# choose the mode alpha coefficient you would like to observe for


time = np.arange(0, tau, tau / 10000)


def Alpha_function_real_1(t, C_index):
    w = radial_freqs[C_index]
    for i in range(S):
        if i * tau / S <= t:
            if t < (i + 1) * tau / S:
                t1 = i * tau / S
                return sum(
                    C1.real[C_index, j] * omega[j] for j in range(i)
                ) + omega_function(t) * LD_parameter[C_index, 0] * (
                    offset * cos(offset * t) * cos(t * w) /
                    (-(offset ** 2) + w ** 2)
                    - offset
                    * cos(offset * t1)
                    * cos(t1 * w)
                    / (-(offset ** 2) + w ** 2)
                    + w * sin(offset * t) * sin(t * w) /
                    (-(offset ** 2) + w ** 2)
                    - w * sin(offset * t1) * sin(t1 * w) /
                    (-(offset ** 2) + w ** 2)
                )


def Alpha_function_imag_1(t, C_index):
    w = radial_freqs[C_index]
    for i in range(S):
        if t >= i * tau / S:
            if t < (i + 1) * tau / S:
                t1 = i * tau / S
                return sum(
                    C1.imag[C_index, j] * omega[j] for j in range(i)
                ) + omega_function(t) * LD_parameter[C_index, 0] * (
                    offset * sin(t * w) * cos(offset * t) /
                    (-(offset ** 2) + w ** 2)
                    - offset
                    * sin(t1 * w)
                    * cos(offset * t1)
                    / (-(offset ** 2) + w ** 2)
                    - w * sin(offset * t) * cos(t * w) /
                    (-(offset ** 2) + w ** 2)
                    + w * sin(offset * t1) * cos(t1 * w) /
                    (-(offset ** 2) + w ** 2)
                )


fig = plt.figure()

for i in np.arange(1, ion_number+1, 1):
    Alpha_array_imag = 0
    Alpha_array_real = 0
    for t in time:
        Alpha_array_real = np.append(
            Alpha_array_real, Alpha_function_real_1(t, i - 1))
        Alpha_array_imag = np.append(
            Alpha_array_imag, Alpha_function_imag_1(t, i - 1))
    axs = fig.add_subplot(2, 3, i)
    axs.plot(Alpha_array_real, Alpha_array_imag)
    # axs.set_xlabel("Alpha real")
    # axs.set_ylabel("Alpha imaginary")
    axs.grid(True)
axmd = fig.add_subplot(2, 3, 6)
plt.imshow(radial_modes, cmap="bwr", vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel("ion number")
plt.ylabel("mode number")
plt.show()


phonon_number = 10

betta = 1/(np.tanh(1/5*np.log(1+1/phonon_number)))
G_1 = np.exp(-sum(Alpha_function_imag_1(0.9999999*tau, a)**2 +
                  Alpha_function_real_1(tau*0.9999999, a)**2 for a in range(ion_number))*betta/2)
G_2 = np.exp(-sum((Alpha_function_imag_1(0.9999999*tau, a)**2 +
                   Alpha_function_real_1(tau*0.9999999, a)**2)*(LD_parameter[a, 1]/LD_parameter[a, 0])**2 for a in range(ion_number))*betta/2)
G_plus = np.exp(-sum((Alpha_function_imag_1(0.9999999*tau, a)**2+Alpha_function_real_1(tau*0.9999999, a)
                ** 2)*(1+LD_parameter[a, 1]/LD_parameter[a, 0])**2 for a in range(ion_number))*betta/2)
G_minus = np.exp(-sum((Alpha_function_imag_1(0.9999999*tau, a)**2+Alpha_function_real_1(tau*0.9999999, a)
                       ** 2)*(1-LD_parameter[a, 1]/LD_parameter[a, 0])**2 for a in range(ion_number))*betta/2)

Fidelity = 1/8*(2+2*(G_1+G_2)+G_plus+G_minus)
print("Fidelity", Fidelity)
