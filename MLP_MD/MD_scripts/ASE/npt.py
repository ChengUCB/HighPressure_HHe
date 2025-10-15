import sys
import torch
import torch.nn as nn

from mace.calculators import MACECalculator

from ase import Atoms
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
from ase import units
from ase.io import read, write
import numpy as np
import time


temperature = XXX  # Temperature in K
pressure = XXX # Pressure in GPa
# Read structure and setup calculator
atoms = read('restart.xyz', '0')
calculator = MACECalculator(
    model_paths='../hhe_train_ist_swa.model', 
    device='cuda', 
    default_dtype="float32"
)
atoms.set_calculator(calculator)

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature * units.kB)

# Setup NPT Berendsen dynamics
dyn = NPTBerendsen(
    atoms,
    timestep=0.2 * units.fs,
    temperature_K=temperature,
    taut=20 * units.fs,
    pressure_au=pressure * 1e4 * units.bar,
    taup=200 * units.fs,
    compressibility_au=4.57e-5 / units.bar
)

def print_energy(a=atoms):
    """Print per-atom potential, kinetic, and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temp = ekin / (1.5 * units.kB)
    etot = epot + ekin
    print(f'Energy per atom: Epot = {epot:.4f} eV  Ekin = {ekin:.4f} eV '
          f'(T={temp:3.0f} K)  Etot = {etot:.4f} eV')

# Attach loggers
dyn.attach(
    MDLogger(
        dyn, atoms,
        f'npt{temperature}_{pressure}GPa_vdW.log',
        header=True, stress=True, peratom=True, mode="w"
    ),
    interval=100
)

def write_frame():
    atoms.write('npt-MACE.xyz', append=True)

dyn.attach(write_frame, interval=500)

# Run MD simulation
n_steps = 600000
for step in range(n_steps):
    print_energy()
    dyn.run(500)
