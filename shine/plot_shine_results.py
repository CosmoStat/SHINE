#!/usr/bin/env python3
"""
SHINE Results Visualization Script

This script loads and visualizes results from a SHINE inference run.
It generates comprehensive diagnostic plots including:
- Observation visualization (image, PSF, noise map)
- Posterior distributions for all parameters
- Corner plot with confidence intervals
- Shear parameter analysis (if applicable)
- Summary statistics
- Trace plots (if MCMC chains present)
- Parameter correlation matrix

Usage:
    python plot_shine_results.py --output my_output/
"""

import argparse
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def setup_plot_style():
    """Configure matplotlib plot style."""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.style.use('seaborn-v0_8-darkgrid')


def load_data(output_dir):
    """Load observation and posterior data from the output directory.
    
    Args:
        output_dir: Path to directory containing observation.npz and posterior.nc
        
    Returns:
        tuple: (obs_data, idata, posterior)
    """
    output_path = Path(output_dir)
    
    # Load observation data
    obs_file = output_path / 'observation.npz'
    if not obs_file.exists():
        raise FileNotFoundError(f"observation.npz not found in {output_dir}")
    obs_data = np.load(obs_file)
    print(f"Observation data loaded from {obs_file}")
    print(f"Available keys: {list(obs_data.keys())}")
    
    # Load posterior estimates
    posterior_file = output_path / 'posterior.nc'
    if not posterior_file.exists():
        raise FileNotFoundError(f"posterior.nc not found in {output_dir}")
    idata = az.from_netcdf(posterior_file)
    posterior = idata.posterior
    print(f"\nPosterior data loaded from {posterior_file}")
    print(f"Dataset structure:")
    print(posterior)
    
    return obs_data, idata, posterior


def plot_observation(obs_data, output_dir):
    """Visualize the observed galaxy image, PSF, and noise map.
    
    Args:
        obs_data: Loaded observation data
        output_dir: Directory to save the plot
    """
    print("\n" + "="*70)
    print("Plotting Observation")
    print("="*70)
    
    image = obs_data.get('image', None)
    psf = obs_data.get('psf', None)
    noise_map = obs_data.get('noise_map', None)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the galaxy image
    if image is not None:
        im1 = axes[0].imshow(image, origin='lower', cmap='viridis')
        axes[0].set_title('Observed Galaxy Image', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X pixel')
        axes[0].set_ylabel('Y pixel')
        plt.colorbar(im1, ax=axes[0], label='Flux')
        axes[0].text(0.02, 0.98, f'Max: {image.max():.2e}\nMin: {image.min():.2e}',
                     transform=axes[0].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot the PSF
    if psf is not None:
        im2 = axes[1].imshow(psf, origin='lower', cmap='hot')
        axes[1].set_title('PSF Model', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X pixel')
        axes[1].set_ylabel('Y pixel')
        plt.colorbar(im2, ax=axes[1], label='Normalized Flux')
    
    # Plot the noise map
    if noise_map is not None:
        if noise_map.ndim == 0:  # Scalar noise
            axes[2].text(0.5, 0.5, f'Uniform Noise\nσ = {float(noise_map):.2e}',
                         ha='center', va='center', fontsize=16,
                         transform=axes[2].transAxes)
            axes[2].set_title('Noise Map', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        else:  # Spatial noise map
            im3 = axes[2].imshow(noise_map, origin='lower', cmap='plasma')
            axes[2].set_title('Noise Map (σ)', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('X pixel')
            axes[2].set_ylabel('Y pixel')
            plt.colorbar(im3, ax=axes[2], label='Noise σ')
    
    output_file = Path(output_dir) / 'observation_visual.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Observation visualization saved to {output_file}")


def plot_posterior_distributions(posterior, param_names, output_dir):
    """Plot posterior distributions for all parameters.
    
    Args:
        posterior: Posterior dataset
        param_names: List of parameter names
        output_dir: Directory to save the plot
    """
    print("\n" + "="*70)
    print("Plotting Posterior Distributions")
    print("="*70)
    
    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        samples = posterior[param].values
        
        if samples.ndim > 1:
            samples = samples.flatten()
        
        ax.hist(samples, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='black')
        
        mean_val = np.mean(samples)
        median_val = np.median(samples)
        std_val = np.std(samples)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                   label=f'Median: {median_val:.4f}')
        
        ax.set_xlabel(f'{param}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{param} Posterior\nσ = {std_val:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    output_file = Path(output_dir) / 'posterior_distributions.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Posterior distributions saved to {output_file}")


def plot_corner(posterior, param_names, output_dir):
    """Create corner plot with confidence intervals.
    
    Args:
        posterior: Posterior dataset
        param_names: List of parameter names
        output_dir: Directory to save the plot
    """
    if len(param_names) <= 1:
        print("\nCorner plot requires at least 2 parameters. Skipping.")
        return
    
    print("\n" + "="*70)
    print("Plotting Corner Plot")
    print("="*70)
    
    try:
        import corner
    except ImportError:
        print("Installing corner package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "corner"])
        import corner
    
    # Prepare data: stack all parameters as columns
    samples_array = np.column_stack([posterior[param].values.flatten() for param in param_names])
    
    # Create corner plot with confidence intervals
    fig = corner.corner(
        samples_array,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],  # 16th, 50th, 84th percentiles (±1σ)
        levels=(0.68, 0.95),  # 68% and 95% confidence intervals
        show_titles=True,
        title_fmt='.4f',
        smooth=1.0,
        plot_datapoints=True,
        plot_density=True,
        fill_contours=True,
        color='steelblue',
        truth_color='red',
        title_kwargs={"fontsize": 11},
    )
    
    plt.suptitle('Corner Plot: Joint & Marginal Distributions',
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_file = Path(output_dir) / 'corner_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Corner plot saved to {output_file}")


def plot_shear_analysis(posterior, param_names, output_dir):
    """Create corner plot for shear parameters (g1, g2) only.
    
    Args:
        posterior: Posterior dataset
        param_names: List of parameter names
        output_dir: Directory to save the plot
    """
    shear_params = [p for p in param_names if 'g1' in p.lower() or 'g2' in p.lower() or 'shear' in p.lower()]
    
    if not shear_params:
        print("\nNo shear parameters found. Skipping shear analysis.")
        return
    
    print("\n" + "="*70)
    print("Plotting Shear Analysis")
    print("="*70)
    print(f"Found shear parameters: {shear_params}")
    
    g1_param = next((p for p in param_names if 'g1' in p.lower()), None)
    g2_param = next((p for p in param_names if 'g2' in p.lower()), None)
    
    if not (g1_param and g2_param):
        print("Could not identify both g1 and g2 parameters. Skipping.")
        return
    
    # Import corner package
    try:
        import corner
    except ImportError:
        print("Installing corner package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "corner"])
        import corner
    
    g1_samples = posterior[g1_param].values.flatten()
    g2_samples = posterior[g2_param].values.flatten()
    
    # Prepare data: stack g1 and g2 as columns
    samples_array = np.column_stack([g1_samples, g2_samples])
    
    # Calculate statistics
    g1_mean = np.mean(g1_samples)
    g1_std = np.std(g1_samples)
    g2_mean = np.mean(g2_samples)
    g2_std = np.std(g2_samples)
    
    # Create corner plot for shear parameters only
    fig = corner.corner(
        samples_array,
        labels=['g1', 'g2'],
        quantiles=[0.16, 0.5, 0.84],  # 16th, 50th, 84th percentiles (±1σ)
        levels=(0.68, 0.95),  # 68% and 95% confidence intervals
        show_titles=True,
        title_fmt='.4f',
        smooth=1.0,
        plot_datapoints=True,
        plot_density=True,
        fill_contours=True,
        color='steelblue',
        truth_color='red',
        title_kwargs={"fontsize": 11},
    )
    
    plt.suptitle('Shear Parameters Corner Plot (g1, g2)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_file = Path(output_dir) / 'shear_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nShear estimates:")
    print(f"  g1 = {g1_mean:.6f} ± {g1_std:.6f}")
    print(f"  g2 = {g2_mean:.6f} ± {g2_std:.6f}")
    print(f"✓ Shear analysis saved to {output_file}")


def print_summary_statistics(posterior, param_names):
    """Print summary statistics for all parameters.
    
    Args:
        posterior: Posterior dataset
        param_names: List of parameter names
    """
    print("\n" + "="*70)
    print("POSTERIOR SUMMARY STATISTICS")
    print("="*70)
    print(f"{'Parameter':<20} {'Mean':<12} {'Std':<12} {'Median':<12} {'95% CI':<20}")
    print("-"*70)
    
    for param in param_names:
        samples = posterior[param].values.flatten()
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        median_val = np.median(samples)
        ci_low = np.percentile(samples, 2.5)
        ci_high = np.percentile(samples, 97.5)
        
        print(f"{param:<20} {mean_val:<12.6f} {std_val:<12.6f} {median_val:<12.6f} [{ci_low:.6f}, {ci_high:.6f}]")
    
    print("="*70)


def plot_trace(posterior, param_names, output_dir):
    """Plot trace plots for MCMC convergence diagnostics.
    
    Args:
        posterior: Posterior dataset
        param_names: List of parameter names
        output_dir: Directory to save the plot
    """
    has_chains = any(dim in posterior.dims for dim in ['chain', 'draw', 'sample'])
    
    if not has_chains:
        print("\nNo chain/draw dimensions found - likely MAP or point estimate.")
        print("Skipping trace plots.")
        return
    
    print("\n" + "="*70)
    print("Plotting Trace Plots")
    print("="*70)
    
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))
    
    if n_params == 1:
        axes = [axes]
    
    for idx, param in enumerate(param_names):
        samples = posterior[param].values
        
        # Trace plot
        if samples.ndim >= 2:
            for chain in range(samples.shape[0]):
                axes[idx].plot(samples[chain], alpha=0.7, label=f'Chain {chain}')
        else:
            axes[idx].plot(samples, alpha=0.7)
        
        axes[idx].set_ylabel(param, fontsize=11)
        axes[idx].set_xlabel('Iteration', fontsize=11)
        axes[idx].set_title(f'{param} - Trace', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        if samples.ndim >= 2 and samples.shape[0] <= 10:
            axes[idx].legend(fontsize=8)
    
    output_file = Path(output_dir) / 'trace_plots.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Trace plots saved to {output_file}")


def plot_correlation_matrix(posterior, param_names, output_dir):
    """Plot parameter correlation matrix.
    
    Args:
        posterior: Posterior dataset
        param_names: List of parameter names
        output_dir: Directory to save the plot
    """
    if len(param_names) <= 1:
        print("\nOnly one parameter - skipping correlation matrix.")
        return
    
    print("\n" + "="*70)
    print("Plotting Correlation Matrix")
    print("="*70)
    
    # Create correlation matrix
    data_matrix = np.column_stack([posterior[param].values.flatten() for param in param_names])
    corr_matrix = np.corrcoef(data_matrix.T)
    
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=12)
    
    # Add correlation values
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                   ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    output_file = Path(output_dir) / 'correlation_matrix.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation matrix saved to {output_file}")


def main():
    """Main function to run the visualization pipeline."""
    parser = argparse.ArgumentParser(
        description='Visualize SHINE inference results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directory containing observation.npz and posterior.nc files (and where plots will be saved)'
    )
    
    args = parser.parse_args()
    
    # Verify output directory exists
    output_dir = Path(args.output)
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)
    
    print("="*70)
    print("SHINE RESULTS VISUALIZATION")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Setup plotting style
    setup_plot_style()
    
    # Load data
    try:
        obs_data, idata, posterior = load_data(output_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Get parameter names
    param_names = list(posterior.data_vars)
    print(f"\nInferred parameters: {param_names}")
    
    # Generate all plots
    plot_observation(obs_data, output_dir)
    plot_posterior_distributions(posterior, param_names, output_dir)
    plot_corner(posterior, param_names, output_dir)
    plot_shear_analysis(posterior, param_names, output_dir)
    print_summary_statistics(posterior, param_names)
    plot_trace(posterior, param_names, output_dir)
    plot_correlation_matrix(posterior, param_names, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll plots saved to: {output_dir.absolute()}")
    print("\nGenerated plots:")
    print("  • observation_visual.png - Observed image, PSF, and noise map")
    print("  • posterior_distributions.png - Posterior distributions for all parameters")
    if len(param_names) > 1:
        print("  • corner_plot.png - Corner plot with confidence intervals")
        print("  • correlation_matrix.png - Parameter correlation matrix")
    if any('g1' in p.lower() or 'g2' in p.lower() for p in param_names):
        print("  • shear_analysis.png - Detailed shear parameter analysis")
    if any(dim in posterior.dims for dim in ['chain', 'draw', 'sample']):
        print("  • trace_plots.png - MCMC trace plots")
    print("\n✓ All visualizations completed successfully!")


if __name__ == '__main__':
    main()
