import io
import numpy as np
from numpy.lib.function_base import delete
from pylion.functions import langevinbath
from scipy.linalg import eigh, eigvalsh
import pylion as pl
import matplotlib.pyplot as plt

# Constants declaration
ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity


def frequency_to_voltage(
    w_z, w_r, ref_ion, RF_frequency, Z0, R0_eff, kappa
):  # frequencies in Hz
    mass_ref = ref_ion["mass"] * amu
    charge_ref = ref_ion["charge"] * ech
    az = (2 * w_z / RF_frequency) ** 2
    qr = (2 * (2 * w_r / RF_frequency) ** 2 + az) ** 0.5
    DC_voltage = (
        az
        * mass_ref
        * Z0 ** 2
        * 4
        * np.pi ** 2
        * RF_frequency ** 2
        / 8
        / charge_ref
        / kappa
    )
    RF_voltage = (
        qr
        * mass_ref
        * R0_eff ** 2
        * 4
        * np.pi ** 2
        * RF_frequency ** 2
        / 2
        / charge_ref
    )
    return (DC_voltage, RF_voltage)  # voltage in volts


def voltage_to_frequency(
    DC_voltage, RF_voltage, ref_ion, RF_frequency, Z0, R0_eff, kappa
):  # voltage in volts
    mass_ref = ref_ion["mass"] * amu
    charge_ref = ref_ion["charge"] * ech
    az = (
        8
        * charge_ref
        * kappa
        * DC_voltage
        / (mass_ref * Z0 ** 2 * 4 * np.pi ** 2 * RF_frequency ** 2)
    )
    qr = (
        2
        * charge_ref
        * RF_voltage
        / (mass_ref * R0_eff ** 2 * 4 * np.pi ** 2 * RF_frequency ** 2)
    )
    w_z = RF_frequency / 2 * (az) ** 0.5
    w_r = RF_frequency / 2 * (0.5 * qr ** 2 - az / 2) ** 0.5
    if (
        np.real(w_z) < 0
        or np.real(w_r) < 0
        or np.imag(w_z) > 1e-10
        or np.imag(w_r) > 1e-10
    ):
        print("Error: Stable confinement wasn't achieved! Change voltages!")
    return (w_z, w_r)  # frequencies in Hz


def axial_hessian_matrix(
    ion_positions,
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
):
    ion_number = ion_positions.shape[0]
    axial_freqs = []
    for i in range(ion_number):
        w_z, w_r = voltage_to_frequency(
            DC_voltage,
            RF_voltage,
            ion_types[ions_order[i]],
            RF_frequency,
            Z0,
            R0_eff,
            kappa,
        )
        axial_freqs.append(w_z)
    axial_freqs = np.array(axial_freqs)
    arg_ref = np.where(ions_order == reference_ion_type)
    ions_mass = np.array([ion_types[x]["mass"] for x in ions_order])
    ions_charge = np.array([ion_types[x]["charge"] for x in ions_order])
    l = (
        (ions_charge[arg_ref][0] * ech) ** 2
        / (
            ions_mass[arg_ref][0]
            * amu
            * 4
            * np.pi
            * eps0
            * (axial_freqs[arg_ref][0] * 2 * np.pi) ** 2
        )
    ) ** (1 / 3)
    energy_norm = ions_mass[arg_ref][0] * (axial_freqs[arg_ref][0] * 2 * np.pi) ** 2
    ion_positions = np.array(ion_positions) / l
    A_matrix = (
        np.diag(ions_mass * (axial_freqs + np.array(tweezer)) ** 2 * 4 * np.pi ** 2)
        / energy_norm
    )
    for i in range(ion_number):
        S = 0
        for j in range(ion_number):
            if j != i:
                A_matrix[i, j] = (
                    -2
                    * ions_charge[i]
                    * ions_charge[j]
                    / np.abs(ion_positions[i] - ion_positions[j]) ** 3
                )
                S += (
                    2
                    * ions_charge[i]
                    * ions_charge[j]
                    / np.abs(ion_positions[i] - ion_positions[j]) ** 3
                )
        A_matrix[i, i] += S
    M_matrix = np.diag((ions_mass) ** (-0.5))
    A_matrix = A_matrix * ions_mass[arg_ref][0]
    return (A_matrix, M_matrix)


def radial_hessian_matrix(
    ion_positions,
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
):
    ion_number = ion_positions.shape[0]
    radial_freqs = []
    axial_freqs = []
    for i in range(ion_number):
        w_z, w_r = voltage_to_frequency(
            DC_voltage,
            RF_voltage,
            ion_types[ions_order[i]],
            RF_frequency,
            Z0,
            R0_eff,
            kappa,
        )
        radial_freqs.append(w_r)
        axial_freqs.append(w_z)
    radial_freqs = np.array(radial_freqs)
    axial_freqs = np.array(axial_freqs)
    arg_ref = np.where(ions_order == reference_ion_type)
    ions_mass = np.array([ion_types[x]["mass"] for x in ions_order])
    ions_charge = np.array([ion_types[x]["charge"] for x in ions_order])
    l = (
        (ions_charge[arg_ref][0] * ech) ** 2
        / (
            ions_mass[arg_ref][0]
            * amu
            * 4
            * np.pi
            * eps0
            * (axial_freqs[arg_ref][0] * 2 * np.pi) ** 2
        )
    ) ** (1 / 3)
    print(l)
    energy_norm = ions_mass[arg_ref][0] * (axial_freqs[arg_ref][0] * 2 * np.pi) ** 2
    ion_positions = np.array(ion_positions) / l
    B_matrix = (
        np.diag(ions_mass * (radial_freqs + np.array(tweezer)) ** 2 * 4 * np.pi ** 2)
        / energy_norm
    )
    for i in range(ion_number):
        S = 0
        for j in range(ion_number):
            if j != i:
                B_matrix[i, j] = (
                    ions_charge[i]
                    * ions_charge[j]
                    / np.abs(ion_positions[i] - ion_positions[j]) ** 3
                )
                S -= (
                    ions_charge[i]
                    * ions_charge[j]
                    / np.abs(ion_positions[i] - ion_positions[j]) ** 3
                )
        B_matrix[i, i] += S
    M_matrix = np.diag((ions_mass) ** (-0.5))
    B_matrix = B_matrix * ions_mass[arg_ref][0]
    return (B_matrix, M_matrix)


def axial_normal_modes(
    ion_positions,
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
):
    axial_hessian, mass_matrix = axial_hessian_matrix(
        ion_positions,
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
    D = mass_matrix.dot(axial_hessian.dot(mass_matrix))
    freq, normal_vectors = eigh(D)
    normal_vectors = -normal_vectors.T
    freq = freq ** 0.5
    return (freq, normal_vectors)


def radial_normal_modes(
    ion_positions,
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
):
    radial_hessian, mass_matrix = radial_hessian_matrix(
        ion_positions,
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
    D = mass_matrix.dot(radial_hessian.dot(mass_matrix))
    freq, normal_vectors = eigh(D)
    normal_vectors = -normal_vectors.T
    freq = freq ** 0.5
    return (freq, normal_vectors)


def simulation_run(
    ion_types, ions_order, ion_number, ions_initial_splitting, trap,
):

    ions_positions_z = (
        np.linspace(-ion_number / 2, ion_number / 2, ion_number)
        * ions_initial_splitting
    )

    ions_positions = (
        np.concatenate(
            [
                (np.random.rand(ion_number) - 0.5) * 1e-6,
                (np.random.rand(ion_number) - 0.5) * 1e-6,
                ions_positions_z,
            ]
        )
        .reshape((3, ion_number))
        .T
    )

    simulation_name = pl.Simulation("normal_modes")
    ions_storage = []

    # Adding ions and trap into simulation
    for i in range(len(ion_types)):
        ion_number_type = np.count_nonzero(ions_order == i)
        ions = pl.placeions(
            ion_types[i],
            ions_positions[np.where(ions_order == i), :]
            .reshape(ion_number_type, 3)
            .tolist(),
        )
        simulation_name.append(ions)

        pseudotrap = pl.linearpaultrap(trap, ions, all=False)
        simulation_name.append(pseudotrap)

    # ions = pl.placeions(ion_types, ions_positions)
    # s.append(ions)
    # s.append(pl.linearpaultrap(trap, ions, all=False))

    # Creation of Langevin bath
    langevinbath = pl.langevinbath(0, 3e-6)
    simulation_name.append(langevinbath)

    # Description of output file
    dump = pl.dump(
        "positions.txt", variables=["id", "mass", "q", "x", "y", "z"], steps=100
    )
    simulation_name.append(dump)

    # Definition of the evolution time
    simulation_name.append(pl.evolve(1e5))

    # Start of the simulation
    simulation_name.remove(langevinbath)

    simulation_name.execute()

    _, data = pl.readdump("positions.txt")

    # Loading of the simulation results
    final_x = data[-1, :, 3]
    final_y = data[-1, :, 4]
    final_z = data[-1, :, 5]
    final_mass = np.round(data[-1, :, 1] / amu).astype("int32")
    final_charge = np.round(data[-1, :, 2] / ech).astype("int32")

    # Cristal order check
    sorted_inds = final_z.argsort()
    final_x = final_x[sorted_inds]
    final_y = final_y[sorted_inds]
    final_z = final_z[sorted_inds]
    final_mass = final_mass[sorted_inds]
    final_charge = final_charge[sorted_inds]
    initial_mass = np.array([ion_types[x]["mass"] for x in ions_order])
    initial_charge = np.array([ion_types[x]["charge"] for x in ions_order])
    mass_error = np.sum((initial_mass - final_mass) ** 2)
    charge_error = np.sum((initial_charge - final_charge) ** 2)
    if (mass_error + charge_error) == 0:
        print("Ion crysal ordering is correct!")
    else:
        raise Exception(
            "Ion crysal ordering has been changed! Try to set different trap potentials or ion positions!"
        )

    # Check if the crystal structure is linear
    max_radial_variation = (
        (np.max(final_x) - np.min(final_x)) ** 2
        + (np.max(final_y) - np.min(final_y)) ** 2
    ) ** 0.5
    min_axial_distance = np.min(final_z[1:] - final_z[:-1])

    if min_axial_distance * 1e-3 > max_radial_variation:
        print("Ion crystal is linear.")
    else:
        raise Exception("Ion crystal is not linear. Mode structure will be incorrect!")

    timestep = simulation_name.attrs["timestep"]

    return timestep


def comprehensive_plot(
    ions_order, data, radial_modes, axial_modes, radial_freqs, axial_freqs, tweezer
):
    fig = plt.figure()
    grid = plt.GridSpec(3, 5, wspace=0.4, hspace=0.3)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    final_z = data[-1, :, 5]
    final_x = data[-1, :, 3]
    final_z = np.sort(final_z)
    final_x = np.sort(final_x)

    tmp = np.max(ions_order) + 1
    color_column = np.linspace(0.1, 0.8, tmp).reshape(tmp, 1)
    color_map_unit = np.hstack((np.zeros((tmp, 1)), color_column, color_column))
    color_map = color_map_unit[ions_order]
    fig.add_subplot(grid[0, 0:])
    plt.scatter(final_z, final_x, c=color_map)
    plt.title("Ion's equilibrium positions")
    plt.xlabel("Ion's z coordinates")
    plt.ylabel("Ion's x coordinates")
    plt.ylim(
        [
            -max(1, 1.2 * np.max(np.abs(data[-1, :, 3]))),
            max(1, 1.2 * np.max(np.abs(data[-1, :, 3]))),
        ]
    )

    ax1 = fig.add_subplot(grid[1:, :2])
    plt.imshow(radial_modes, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    # for i in range(radial_modes.shape[0]):
    #     for j in range(radial_modes.shape[0]):
    #         ax1.text(j, i, np.round(radial_modes[i, j], 4), ha="center", va="center")
    plt.xlabel("ion number")
    plt.ylabel("mode number")
    plt.tight_layout()

    fig.add_subplot(grid[1:, 2])
    plt.plot([], [], color="red", label="radial", linewidth=0.5)
    plt.plot([], [], color="blue", label="axial", linewidth=0.5)

    for omega in radial_freqs:
        plt.plot([-1, 0], [omega, omega], color="red", linewidth=0.5)
    for omega in axial_freqs:
        plt.plot([-1, 0], [omega, omega], color="blue", linewidth=0.5)

    plt.ylabel("$\omega/\omega_{\mathrm{com}}^{\mathrm{ax}}$")
    plt.xticks([])
    plt.xlim(-1, 2)
    plt.legend(loc="upper right")
    plt.tight_layout()

    fig.add_subplot(grid[1:, 3:])
    plt.imshow(axial_modes, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel("ion number")
    plt.ylabel("mode number")
    plt.tight_layout()

    plt.savefig("Normal modes structure", dpi=300)
    plt.show()


def relation_plot(modes):
    relation = np.zeros((np.shape(modes)[0], np.shape(modes)[0] - 1))
    for i in range(np.shape(modes)[0]):
        for j in range(4):
            relation[i, j] = np.abs(
                min((modes[i, 4]), modes[i, j], key=abs)
                / max(modes[i, j], np.abs(modes[i, 4]), key=abs)
            )
    arg = np.where(relation == relation.max())
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(relation)
    ax1.text(
        arg[1][0], arg[0][0], np.round(np.max(relation), 4), ha="center", va="center"
    )
    fig.colorbar(im1, orientation="vertical", ax=ax1)
    ax1.set_title("Relations between ancilla and Ca40 modes")
    ax1.set_ylabel("Mode number")
    ax1.set_xlabel("Ca ion number except for ancilla")
    im2 = ax2.imshow(modes / np.max(np.abs(modes)), vmin=-1, vmax=1)
    ax2.set_ylabel("Mode number")
    ax2.set_xlabel("Ca ion number")
    fig.colorbar(im2, orientation="vertical", ax=ax2)
    plt.show()


def get_modes(
    ion_types,
    ions_order,
    ion_number,
    ions_initial_splitting,
    trap,
    tweezer,
    reference_ion_type,
):
    timestep = simulation_run(
        ion_types, ions_order, ion_number, ions_initial_splitting, trap
    )
    _, data = pl.readdump("positions.txt")
    final_z = data[-1, :, 5]
    final_z = np.sort(final_z)
    axial_freqs, axial_modes = axial_normal_modes(
        final_z,
        trap["endcapvoltage"],
        trap["voltage"],
        ion_types,
        ions_order,
        trap["frequency"],
        trap["length"],
        trap["radius"],
        trap["kappa"],
        tweezer,
        reference_ion_type,
    )
    radial_freqs, radial_modes = radial_normal_modes(
        final_z,
        trap["endcapvoltage"],
        trap["voltage"],
        ion_types,
        ions_order,
        trap["frequency"],
        trap["length"],
        trap["radius"],
        trap["kappa"],
        tweezer,
        reference_ion_type,
    )
    # comprehensive_plot(
    #     ions_order, data, radial_modes, axial_modes, radial_freqs, axial_freqs, tweezer
    # )

    return (radial_modes, radial_freqs, axial_modes, axial_freqs)

