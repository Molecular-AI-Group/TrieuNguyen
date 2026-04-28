import sys
import json 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

def plot_best_smiles_progression(df, output_path, mols_per_row=4):
    """
    Plot the progression of best SMILES molecules, only showing when a new best is found.
    
    Args:
        df: DataFrame with columns [Step, SMILES, HasSubstructure, LogP, NumHeavyAtoms, MW, Score, Penalized Score]
        output_path: Path to save the output image
        mols_per_row: Number of molecules per row (max 4)
    """
    # Sort by step
    df_sorted = df.sort_values('Step')
    
    # Track best score and collect progression
    best_score = -float('inf')
    best_mols = []
    
    for step, group in df_sorted.groupby('Step'):
        # Get the best molecule in this step
        best_in_step = group.loc[group['Score'].idxmax()]
        
        # Only record if it's better than previous best
        if best_in_step['Score'] > best_score:
            best_score = best_in_step['Score']
            best_mols.append(best_in_step)
    
    if not best_mols:
        print("No valid molecules found to plot")
        return
    
    # Convert to DataFrame for easier access
    df_best = pd.DataFrame(best_mols)
    
    # Create molecule objects and labels
    mols = []
    labels = []
    for idx, row in df_best.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:
            mols.append(mol)
            label = (f"Step: {int(row['Step'])}\n"
                    f"Score: {row['Score']:.3f}\n"
                    f"LogP: {row['LogP']:.2f}\n"
                    f"Heavy Atoms: {int(row['NumHeavyAtoms'])}\n"
                    f"MW: {row['MW']:.1f}")
            labels.append(label)
    
    if not mols:
        print("No valid molecules found to plot")
        return
    
    # Create image grid
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(400, 400),
        legends=labels,
        returnPNG=False
    )
    
    # Save image
    img.save(output_path)
    print(f"Saved plot to {output_path} with {len(mols)} molecules")


def plot_property_distributions(df, config, output_path):
    """
    Plot distributions of molecular properties (LogP, MW, Heavy Atoms, SA) for all generated molecules
    and for molecules with substructure match, with vertical lines indicating input values.
    
    Args:
        df: DataFrame with columns [Step, SMILES, HasSubstructure, LogP, NumHeavyAtoms, MW, SA, Score, Penalized Score]
        config: Dictionary containing configuration including input molecule properties
        output_path: Path to save the output image
    """
    # Filter valid molecules (exclude None values)
    df_valid = df.dropna(subset=['LogP', 'MW', 'NumHeavyAtoms', 'SA'])
    
    # Filter molecules with substructure match
    df_match = df_valid[df_valid['HasSubstructure'] == True]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    properties = [
        ('LogP', config['input_logp'], 'LogP'),
        ('MW', config['input_mw'], 'Molecular Weight'),
        ('NumHeavyAtoms', config['input_num_heavy_atoms'], 'Number of Heavy Atoms'),
        ('SA', config['input_sa'], 'Synthetic Accessibility')
    ]
    
    # Top row: All generated molecules
    for idx, (prop, input_val, label) in enumerate(properties):
        ax = axes[0, idx]
        data = df_valid[prop].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(input_val, color='red', linestyle='--', linewidth=2, label=f'Input: {input_val:.2f}')
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{label} - All', fontsize=12, fontweight='bold')
            ax.legend()
            # ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
    
    # Bottom row: Only molecules with substructure match
    for idx, (prop, input_val, label) in enumerate(properties):
        ax = axes[1, idx]
        data = df_match[prop].dropna()
        
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(input_val, color='red', linestyle='--', linewidth=2, label=f'Input: {input_val:.2f}')
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{label} - Has Substructure', fontsize=12, fontweight='bold')
            ax.legend()
            # ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No matches found', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved property distribution plot to {output_path}")
    print(f"Total valid molecules: {len(df_valid)}")
    print(f"Molecules with substructure match: {len(df_match)}")

def plot_action_distribution(actions_path, output_path, num_steps=2000, dimension=128):
    """
    Plot the distribution of actions over time.
    
    Args:
        actions_path: Path to the numpy file containing actions
        output_path: Path to save the output image
        num_steps: Number of optimization steps
        dimension: Dimensionality of the action space
    """
    import matplotlib.pyplot as plt
    
    # Load actions
    actions = np.load(actions_path)  # Shape: (num_steps * num_samples, dimension)
    
    # Reshape to (num_steps, num_samples, dimension)
    num_samples = actions.shape[0] // num_steps
    actions = actions.reshape(num_steps, num_samples, dimension)
    
    # Compute mean action per step (average over samples)
    mean_actions = actions.mean(axis=1)  # Shape: (num_steps, dimension)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(mean_actions.T, aspect='auto', cmap='RdBu_r', 
                   origin='lower', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Dimension', fontsize=14)
    ax.set_title('Action Distribution Over Time (Mean per Step)', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Action Value', fontsize=12)
    
    # Adjust ticks for better readability
    ax.set_xticks(np.linspace(0, num_steps-1, 11))
    ax.set_xticklabels([int(x) for x in np.linspace(0, num_steps-1, 11)])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved action distribution plot to {output_path}")
    print(f"Action shape: {actions.shape}")
    print(f"Action mean: {actions.mean():.4f}, std: {actions.std():.4f}")

def plot_umap_molecules(df, output_path, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Extract unique molecules with substructure match, compute UMAP embeddings, and plot colored by score.
    Left plot: all molecules. Right plot: molecules with trajectory of best SMILES at each step.
    
    Args:
        df: DataFrame with columns [Step, SMILES, HasSubstructure, LogP, NumHeavyAtoms, MW, SA, Score, Penalized Score]
        output_path: Path to save the output image
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: UMAP distance metric
    """
    from rdkit.Chem import AllChem
    from umap import UMAP
    from matplotlib.lines import Line2D
    
    # Get unique molecules with their best scores
    df_unique = df.groupby('SMILES').agg({
        'Score': 'max',
        'HasSubstructure': 'first',
        'Step': 'first'  # Keep track of when it first appeared
    }).reset_index()
    
    # Filter out invalid molecules and keep only those with substructure match
    df_unique = df_unique[df_unique['SMILES'].notna()].copy()
    df_unique = df_unique[df_unique['HasSubstructure'] == True].copy()
    
    if len(df_unique) == 0:
        print("No molecules with substructure match found")
        return
    
    # Generate molecular fingerprints
    fps = []
    valid_rows = []
    
    for idx, row in df_unique.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(np.array(fp))
            valid_rows.append(idx)
    
    if len(fps) < 2:
        print("Not enough valid molecules with substructure match for UMAP plotting")
        return
    
    # Filter DataFrame to only valid molecules
    df_valid = df_unique.loc[valid_rows].reset_index(drop=True)
    fps_array = np.array(fps)
    
    # Compute UMAP embedding
    reducer = UMAP(n_neighbors=min(n_neighbors, len(fps)-1), min_dist=min_dist, metric=metric, random_state=42)
    embedding = reducer.fit_transform(fps_array)
    
    # Get trajectory of best SMILES at each step
    df_sorted = df.sort_values('Step')
    best_smiles_trajectory = []
    best_score = -float('inf')
    
    for step, group in df_sorted.groupby('Step'):
        best_in_step = group.loc[group['Score'].idxmax()]
        if best_in_step['Score'] > best_score and best_in_step['HasSubstructure']:
            best_score = best_in_step['Score']
            best_smiles_trajectory.append(best_in_step['SMILES'])
    
    # Find indices of trajectory molecules in df_valid
    trajectory_indices = []
    trajectory_steps = []
    for smiles in best_smiles_trajectory:
        idx = df_valid[df_valid['SMILES'] == smiles].index
        if len(idx) > 0:
            trajectory_indices.append(idx[0])
            # Get the step when this became the new best
            step = df_sorted[df_sorted['SMILES'] == smiles]['Step'].min()
            trajectory_steps.append(step)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: All molecules with substructure match
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=df_valid['Score'], 
        cmap='viridis', 
        s=80, 
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    ax1.set_xlabel('UMAP 1', fontsize=14)
    ax1.set_ylabel('UMAP 2', fontsize=14)
    ax1.set_title('UMAP of Molecules with Substructure Match', fontsize=16, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Score', fontsize=14)
    
    # Right plot: Same but with trajectory
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=df_valid['Score'], 
        cmap='viridis', 
        s=80, 
        alpha=0.4,
        edgecolors='black',
        linewidth=0.2
    )
    
    # Plot trajectory if exists
    if len(trajectory_indices) > 1:
        trajectory_coords = embedding[trajectory_indices]
        
        # Plot line connecting trajectory points with gradient
        for i in range(len(trajectory_coords) - 1):
            ax2.plot(trajectory_coords[i:i+2, 0], trajectory_coords[i:i+2, 1], 
                    'k-', linewidth=2, alpha=0.6, zorder=10)
        
        # Plot trajectory points colored by step order
        colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory_coords)))
        ax2.scatter(trajectory_coords[:, 0], trajectory_coords[:, 1], 
                   s=200, c=colors, edgecolors='black', 
                   linewidth=2, zorder=15, alpha=0.9)
        
        # Add step numbers on trajectory points
        for i, (coord, step) in enumerate(zip(trajectory_coords, trajectory_steps)):
            ax2.annotate(f'{int(step)}', 
                        xy=coord, 
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                        zorder=20)
        
        # Mark start and end with special symbols
        ax2.scatter(trajectory_coords[0, 0], trajectory_coords[0, 1], 
                   marker='^', s=400, c='lime', edgecolors='black', 
                   linewidth=3, zorder=25, label='Start')
        
        ax2.scatter(trajectory_coords[-1, 0], trajectory_coords[-1, 1], 
                   marker='*', s=600, c='gold', edgecolors='black', 
                   linewidth=3, zorder=25, label='Best')
        
        ax2.legend(loc='best', fontsize=12, framealpha=0.9)
    
    ax2.set_xlabel('UMAP 1', fontsize=14)
    ax2.set_ylabel('UMAP 2', fontsize=14)
    ax2.set_title('UMAP with Best Molecule Trajectory', fontsize=16, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Score', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved UMAP plot to {output_path}")
    print(f"Molecules with substructure match: {len(df_valid)}")
    print(f"Score range: [{df_valid['Score'].min():.3f}, {df_valid['Score'].max():.3f}]")
    print(f"Trajectory length: {len(trajectory_indices)} molecules")


if __name__ == '__main__': 
    folder = sys.argv[1]
    df = pd.read_pickle(f"{folder}/results.pkl")  # Load from pickle to get embeddings
    with open(f"{folder}/config.json", "r") as f:
        config = json.load(f)
    
    # Plot best SMILES progression
    plot_best_smiles_progression(df, f"{folder}/best_smiles_progression.png")
    
    # Plot property distributions
    plot_property_distributions(df, config, f"{folder}/property_distributions.png")

    # Plot action distribution over time
    plot_action_distribution(
        f"{folder}/actions.npy", 
        f"{folder}/action_distribution.png",
        num_steps=config['num_steps'],
        dimension=config['dimension']
    )
    
    # Plot UMAP of unique molecules with substructure match (fingerprint-based)
    plot_umap_molecules(df, f"{folder}/umap_molecules_fingerprint.png")
