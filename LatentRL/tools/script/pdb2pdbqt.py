import argparse
import subprocess

parser = argparse.ArgumentParser(description="Convert a PDB file to PDBQT format using AutoDockTools.")
parser.add_argument("-l", "--ligand", required=True, help="Path to the input PDB file.")
parser.add_argument("-o", "--output", default="output.pdbqt", help="Path to the output PDBQT file (default: output.pdbqt).")
args = parser.parse_args()


def prepare_ligand(input_pdb, output_pdbqt):
    result = subprocess.run(
        ["/home/80027464/LatentRL/tools/AutoDock/mgltools_x86_64Linux2_1.5.7/bin/pythonsh",
         "/home/80027464/LatentRL/tools/AutoDock/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
         "-l", input_pdb,
         "-o", output_pdbqt]
    )
    return result

def main():
    result = prepare_ligand(args.ligand, args.output)
    if result.returncode == 0:
        print(f"Successfully converted {args.ligand} to {args.output}")
    else:
        print(f"Error converting {args.ligand} to {args.output}. Return code: {result.returncode}")

if __name__ == "__main__":
    main()