import time 
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run AutoDock Vina docking.")
parser.add_argument("-l", "--ligand", required=True, help="Path to the input PDBQT ligand file.")
parser.add_argument("-r", "--receptor", default="/home/80027464/LatentRL/tools/AutoDock/mols/protein_no_MG.pdbqt", help="Path to the input PDBQT receptor file.")
parser.add_argument("-c", "--config", default="/home/80027464/LatentRL/tools/AutoDock/mols/config_ori_25.txt", help="Path to the Vina configuration file.")
parser.add_argument("-o", "--output", default="test_out.pdbqt", help="Path to the output PDBQT file (default: test_out.pdbqt).")
parser.add_argument("-g", "--log", default="test_out.log", help="Path to the log file (default: test_out.log).")
args = parser.parse_args()

b = time.time()
result = subprocess.run(
    ["/home/80027464/LatentRL/tools/AutoDock/autodock_vina_1_1_2_linux_x86/bin/vina",
     "--ligand", args.ligand,
     "--receptor", args.receptor,
     "--config", args.config,
     "--out", args.output, "--log", args.log]
)
print(time.time() - b)

with open(args.log, "r") as f:
    lines = f.readlines()

results = {}
for line in lines:
    parts = line.strip().split()
    if len(parts) == 4 and parts[0].isdigit():
        mode = int(parts[0])
        results[mode] = {
            "affinity": float(parts[1]),
            "rmsd_lb": float(parts[2]),
            "rmsd_ub": float(parts[3])
        }
# print(results)