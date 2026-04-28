from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
import argparse
from tqdm import tqdm


def canonicalize(x):
    try:
        mol = Chem.MolFromSmiles(x)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def chunked_iterable(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main(input_file, output_file, n_jobs, chunk_size):
    # Count total lines first (for accurate tqdm)
    with open(input_file) as f:
        total = sum(1 for _ in f)

    with open(input_file) as fin, open(output_file, "w") as fout:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            pbar = tqdm(total=total, desc="Canonicalizing")

            for chunk in chunked_iterable(
                (line.strip() for line in fin if line.strip()),
                chunk_size
            ):
                results = executor.map(canonicalize, chunk)
                for r in results:
                    fout.write(f"{r}\n")
                    pbar.update(1)

            pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()

    main(args.input, args.output, args.n_jobs, args.chunk_size)
