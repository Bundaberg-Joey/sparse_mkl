import os
from glob import glob
from subprocess import run
from shutil import copytree, copy

import numpy as np
from ase.io import read


CUTOFF = 16.0

class RaspaOutputNotExist(FileNotFoundError):
  pass


def find_minimum_image(cell, cutoff):
    ncutoff = cutoff + 1e-8 * cutoff
    V = np.abs(np.linalg.det(cell))
    a, b, c = cell
    Xa = np.cross(b, c)
    ha = V / np.linalg.norm(Xa)
    na = int(np.ceil(2 * ncutoff / ha))
    Xb = np.cross(a, c)
    hb = V / np.linalg.norm(Xb)
    nb = int(np.ceil(2 * ncutoff / hb))
    Xc = np.cross(a, b)
    hc = V / np.linalg.norm(Xc)
    nc = int(np.ceil(2 * ncutoff / hc))
    return na, nb, nc


def xe_kr_input(cif_file, na, nb, nc):
    sim_details = F"""SimulationType                MonteCarlo
NumberOfCycles                10
NumberOfInitializationCycles  10
Restart File                  no
ChargeMethod                  none
CutOff                        16.0

Framework 0
FrameworkName {cif_file}
UnitCells {na} {nb} {nc}
ExternalTemperature 273
ExternalPressure 1e6
RemoveAtomNumberCodeFromLabel yes

Component 0 MoleculeName        xenon
ChargeMethod                    None
IdealGasRosenbluthWeight        1.0
FugacityCoefficient             0.9253
MoleculeDefinition              local
MolFraction                     0.20
IdentityChangeProbability       1.0
  NumberOfIdentityChanges       2
  IdentityChangesList           0 1
TranslationProbability          1.0
ReinsertionProbability          1.0
SwapProbability                 1.0
CreateNumberOfMolecules         0

Component 1 MoleculeName        krypton
ChargeMethod                    None
IdealGasRosenbluthWeight        1.0
FugacityCoefficient             0.9671
MoleculeDefinition              local
MolFraction                     0.80
IdentityChangeProbability       1.0
  NumberOfIdentityChanges       2
  IdentityChangesList           0 1
TranslationProbability          1.0
ReinsertionProbability          1.0
SwapProbability                 1.0
CreateNumberOfMolecules         0
    """
    return sim_details

def write_sim_files(sim_dir, cif_name, na, nb, nc):
    # writes simulation.input along with all other files needed for the raspa sim to a single dir
    # returns the name of the directoy containing all the files + cif file
    sim_details = xe_kr_input(cif_name.replace('.cif', ''), na, nb, nc)
    copy(os.path.join('cifs', cif_name), os.path.join(sim_dir, cif_name))
    sim_file_name = F"simulation_{cif_name.replace('.cif', '')}.input"
    with open(os.path.join(sim_dir, sim_file_name), 'w') as f:
        f.writelines(sim_details)
    return sim_file_name


def parse_output(results_dir, cif_name_clean):
    path = os.path.join(results_dir, 'Output', 'System_0')
    try:
        base_path = glob(F"{path}/output_{cif_name_clean}_*.data")[0]
    except IndexError:
        raise RaspaOutputNotExist

    components = {}
    with open(base_path, 'r') as fd:
        for line in fd:
            if "Number of molecules:" in line:
                break
        for line in fd:
            if line.startswith("Component"):
                name = line.split()[-1][1:-1]
            if "Average loading absolute   " in line:
                res = float(line.split(" +/-")[0].split()[-1])
                components[name] = res
    return components


class RaspaRegistry:
    
    def __init__(self, cif_list_path, simulation_dir='raspa_dir'):
        self.simulation_dir = str(simulation_dir)
                
        with open(str(cif_list_path), 'r') as f:
            self.cifs = [i.strip() for i in f.readlines()]
            
        if not os.path.exists(self.simulation_dir):
            copytree('simulation_template', self.simulation_dir)
            
    def run_simulation(self, idx):
        cif_name = self.cifs[idx]
        cif_name_clean = cif_name.replace('.cif', '')
        
        atoms = read(os.path.join('cifs', cif_name), format="cif")
        cell = np.array(atoms.cell)
        na, nb, nc = find_minimum_image(cell, CUTOFF)   # 1. get number of unit cells to use
        sim_file_name = write_sim_files(self.simulation_dir, cif_name, na, nb, nc)  # 2. write files to location
        run(["simulate", "-i", sim_file_name], cwd=self.simulation_dir)  # 3. run simulation
        components = parse_output(self.simulation_dir, cif_name_clean)  # 4. extract results
            
        selectivity = np.log(1 + (4 * components['xenon'])) - np.log(1 + components['krypton']) # 5. calc selectivity
        results = {'index': idx, 'name': cif_name, 'selectivity': selectivity, **components}
        return results  # 6. return selectivity along with cif name
    
    def __len__(self):
        return len(self.cifs)
