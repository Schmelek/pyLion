from turtle import shape
import pylion as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, minimize
from sympy import rad
import get_modes
from numpy import complex128, sin as sin
from numpy import cos as cos
from scipy.linalg import eig
import time
from mpl_toolkits.mplot3d.axes3d import Axes3D


# Constants declaration
ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity
c = 299792458
hbar = 1.054571817e-34

# use filename for simulation name
# name = Path(__file__).stem


# Declaration of ion types used in the simulation
ion_types = [{"mass": 171, "charge": 1}, {"mass": 138, "charge": 1}]
# Ions ordering. You should place ion type number from the previous list in a desired order.
ions_order = [0]*4+[0]

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
w_z = 1.6e5  # axial secular frequency in Hz
w_r = 3.06e6  # radial secular frequency in Hz

tw = np.sqrt((171/138)**2-1)*w_r
tweezer = [0]+[0]*4
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
N = ion_number


# if you want to use (py)Lion
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

# if you want to use regular python

# w = np.array([w_z]*N)*2*np.pi
# m = np.array([171*amu]*N)
# k = ech**2/(4*np.pi*eps0)
# A = m*w**2
# B = np.array([5e-3]*N)


# def equations(z):
#     z = np.array(z)
#     out = np.empty(N)
#     for i in range(N):
#         out[i] = A[i]*z[i] - k*sum((z[i]-z[j])/(np.abs(z[i]-z[j])**3)
#                                     for j in range(N) if j != i)
#     return out

# final_z = fsolve(equations, np.linspace(-1e-5, 1e-5, N))


radial_freqs = np.array([2.9867, 3.0127, 3.0344, 3.0522, 3.0692])*1e6*2*np.pi
axial_freqs = axial_freqs*w_z*2*np.pi

P = 9
offset = 3.044e6*2*np.pi
tau = 235e-6

S = 1000


def fidelity(offset, tau, P, tweezer):

    LD_parameter = np.zeros((ion_number, 2))
    ions = [0, 4]
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

    C1 = np.zeros((N, P), dtype=complex128)
    for i in range(N):
        for j in range(P):
            t1 = j * tau / P
            t2 = (j + 1) * tau / P
            w = radial_freqs[i]
            C1[i, j] = (
                -offset * cos(offset * t1) * cos(t1 * w) /
                (-(offset ** 2) + w ** 2)
                + offset * cos(offset * t2) * cos(t2 * w) /
                (-(offset ** 2) + w ** 2)
                - w * sin(offset * t1) * sin(t1 * w) /
                (-(offset ** 2) + w ** 2)
                + w * sin(offset * t2) * sin(t2 * w) /
                (-(offset ** 2) + w ** 2)
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

    phonon_number = 10
    betta = 1/(np.tanh(1/2*np.log(1+1/phonon_number)))
    B = 1/4*sum(np.dot(np.conjugate(np.reshape(C1[k, :], (P, 1))), np.reshape(C1[k, :], (1, P)))*(
        1+(LD_parameter[k, 1]/LD_parameter[k, 0])**2) for k in range(N))*betta
    D = np.zeros((P, P))
    for n in range(P):
        t1p = n * tau / P
        t2p = (n + 1) * tau / P
        t1 = n * tau / P

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
        D[n, n] = sum(
            LD_parameter[a, 0] * LD_parameter[a, 1] * tmp(radial_freqs[a])
            for a in range(ion_number))
    for n in range(P):
        for m in range(n):
            t1 = n * tau / P
            t2 = (n + 1) * tau / P
            t1p = m * tau / P
            t2p = (m + 1) * tau / P

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
            D[n, m] = sum(
                tmp(radial_freqs[a]) * LD_parameter[a, 0] * LD_parameter[a, 1]
                for a in range(ion_number)
            )
            D[m, n] = D[n, m]

    F = B+B.T
    V = D+D.T
    eigvals, eigvectors = eig(F, V)
    for i in range(P):
        if np.dot(eigvectors[:, i], (np.dot(D, eigvectors[:, i]))) < 0:
            eigvectors[:, i] = eigvectors[:, i] * \
                (-np.pi * 7 / (4 *
                               np.dot(eigvectors[:, i], (np.dot(D, eigvectors[:, i]))))) ** 0.5
        else:
            eigvectors[:, i] = eigvectors[:, i] * \
                (np.pi / (4 * np.dot(eigvectors[:, i],
                                     (np.dot(D, eigvectors[:, i]))))) ** 0.5

    alpha = np.dot(C1, eigvectors)

    betta = 1/(np.tanh(1/2*np.log(1+1/phonon_number)))
    fidelity = np.zeros(P)
    for i in range(P):
        G_1 = np.exp(-sum(np.abs(alpha[k, i]) **
                          2 for k in range(ion_number))*betta/2)
        G_2 = np.exp(-sum(np.abs(alpha[k, i]*(LD_parameter[k, 1] /
                                              LD_parameter[k, 0]))**2 for k in range(ion_number))*betta/2)
        G_plus = np.exp(-sum(np.abs(alpha[k, i]*(1+LD_parameter[k, 1] /
                        LD_parameter[k, 0]))**2 for k in range(ion_number))*betta/2)
        G_minus = np.exp(-sum(np.abs(alpha[k, i]*(1-LD_parameter[k, 1] /
                                                  LD_parameter[k, 0]))**2 for k in range(ion_number))*betta/2)
        fidelity[i] = 1/8*(2+2*(G_1+G_2)+G_plus+G_minus)

    omega = eigvectors[:, np.argmax(fidelity)]

    def omega_function(t): return np.piecewise(
        t, [((i * tau / P <= t) & (t < (i + 1) * tau / P))
            for i in range(P)], omega
    )

    def Alpha_function_imag_1(t, C_index):
        w = radial_freqs[C_index]
        for i in range(P):
            if t >= i * tau / P:
                if t < (i + 1) * tau / P:
                    t1 = i * tau / P
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

    def Alpha_function_real_1(t, C_index):
        w = radial_freqs[C_index]
        for i in range(P):
            if i * tau / P <= t:
                if t < (i + 1) * tau / P:
                    t1 = i * tau / P
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
    time = np.arange(0, tau, tau / 10000)
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
    return omega, np.max(fidelity)


omega_matrix = np.zeros((P, S))
start_time = time.time()
fidelity_matrix = np.zeros(S)
omega, fidel = fidelity(offset, tau, P, tweezer)
print(fidel)
tweezer_array = np.linspace(tw-1e6, tw+1e6, S)
offset_array = np.linspace(3.0e6*2*np.pi, 3.9e6*2*np.pi, S)
tau_array = np.linspace(100e-6, 300e-6, S)
offset = 3795495.4954954954*2*np.pi
# for i in range(S):
#     # for j in range(S):
#     omega_matrix[:, i], fidelity_matrix[i] = fidelity(
#         offset,  tau, P, tweezer_array[i])
# print("--- %s seconds ---" % (time.time() - start_time))

# tmp = np.zeros(S)
# for i in range(S):
#     tmp[i] = np.max(np.abs(omega_matrix[:, i]))
# ind = np.where((tmp/(2*np.pi) < 9e5) & (tmp/(2*np.pi) > 1e5))[0]


# omega, fid = fidelity(offset,  tau, P, tweezer_array[ind[np.argmax(
#     fidelity_matrix[ind])]])
# tweezer = [tweezer_array[ind[np.argmax(
#     fidelity_matrix[ind])]]] + [0]*4
# radial_freqs, radial_modes = get_modes.radial_normal_modes(final_z,
#                                                            trap["endcapvoltage"],
#                                                            trap["voltage"],
#                                                            ion_types,
#                                                            ions_order,
#                                                            trap["frequency"],
#                                                            trap["length"],
#                                                            trap["radius"],
#                                                            trap["kappa"],
#                                                            tweezer,
#                                                            reference_ion_type)
# axial_freqs, axial_modes = get_modes.axial_normal_modes(final_z,

#                                                         trap["endcapvoltage"],
#                                                         trap["voltage"],
#                                                         ion_types,
#                                                         ions_order,
#                                                         trap["frequency"],
#                                                         trap["length"],
#                                                         trap["radius"],
#                                                         trap["kappa"],
#                                                         tweezer,
#                                                         reference_ion_type)


# print("Fidelity", fid)
# print("Radial frequencies", radial_freqs/(2*np.pi))
# print("Offset", offset/(2*np.pi))
# print("Omega", omega/(2*np.pi))
# print("Tweezer frequency", tweezer_array[ind[np.argmax(
#     fidelity_matrix[ind])]])


plt.bar(np.arange(P), omega/np.max(omega), color="b")
plt.ylabel("Rabi frequency")
plt.xlabel("Segment number")
plt.grid()
plt.show()

# plt.plot(tweezer_array, -np.log(1-fidelity_matrix))
# plt.axvline(x=tw, color='r')
# plt.xlabel("Tweezer frequency, Hz")
# plt.ylabel("-ln(1-Fidelity)")
# plt.grid()
# plt.show()

# plt.plot(tweezer_array, tmp)
# plt.axvline(x=tw, color='r')
# plt.xlabel("Tweezer frequency, Hz")
# plt.ylabel("Maximum Rabi frquency")
# plt.grid()
# plt.show()

# get_modes.comprehensive_plot(
#     ions_order, data, radial_modes, axial_modes, radial_freqs, axial_freqs, tweezer)

# print("Fidelity without tweezer", fidelity(offset, tau, P, 0)[1])
# print(tw)
# plt.bar(np.arange(P), fidelity(offset, tau, P, 0)[0]/(2*np.pi), color="b")
# plt.xlabel("Rabi frequency")
# plt.ylabel("Segment number")
# plt.grid()
# plt.show()
# radial_freqs, radial_modes = get_modes.radial_normal_modes(final_z,
#                                                            trap["endcapvoltage"],
#                                                            trap["voltage"],
#                                                            ion_types,
#                                                            ions_order,
#                                                            trap["frequency"],
#                                                            trap["length"],
#                                                            trap["radius"],
#                                                            trap["kappa"],
#                                                            [0]*5,
#                                                            reference_ion_type)
# axial_freqs, axial_modes = get_modes.axial_normal_modes(final_z,

#                                                         trap["endcapvoltage"],
#                                                         trap["voltage"],
#                                                         ion_types,
#                                                         ions_order,
#                                                         trap["frequency"],
#                                                         trap["length"],
#                                                         trap["radius"],
#                                                         trap["kappa"],
#                                                         [0]*5,
#                                                         reference_ion_type)

# get_modes.comprehensive_plot(
#     ions_order, data, radial_modes, axial_modes, radial_freqs, axial_freqs, [0]*5)
