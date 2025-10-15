import sys
import numpy as np
from ase import Atoms
from ase.io import read,write

pressure_est = [100, 150, 200, 400, 600, 800,1000] # in GPa
num_atoms = 128 
num_H_atoms_list = [2, 6, 10, 18, 26, 34, 50, 58, 66, 70, 74, 78, 82, 86, 90, 94, 98, 100, 102, 104, 106, 108, 110, 114, 118, 122, 126]


for Pressure_Gpa in pressure_est:
    for num_H_atoms in num_H_atoms_list:
        num_He_atoms = num_atoms - num_H_atoms
        min_H_distance = 0.67
        min_He_distance = 0.8
        min_distance = 0.7

        # v=A^3/atom
        # v*(p_GPa)**(1/2) ~ 26

        box_size = (num_atoms * (26.5/(Pressure_Gpa**0.5)))**(1./3.)*(1 + 0.24*num_He_atoms/num_atoms) - min_distance
        positions = []
        symbols = []

        while len(positions) < num_H_atoms:
            new_pos = np.random.rand(3) * box_size
            if all(np.linalg.norm(new_pos - p) >= min_H_distance for p in positions):
                positions.append(new_pos)
                symbols.append("H")

        while len(positions) < num_H_atoms + num_He_atoms:
            new_pos = np.random.rand(3) * box_size
            if all(np.linalg.norm(new_pos - p) >= min_He_distance for p in positions):
                positions.append(new_pos)
                symbols.append("He")

        atoms = Atoms(symbols, positions=positions, cell=[box_size + min_distance, box_size + min_distance, box_size + min_distance], pbc=True)

        write('H-'+str(num_H_atoms)+'-Pest-'+str(Pressure_Gpa)+'.xyz', atoms, format='xyz')

