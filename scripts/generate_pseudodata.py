#!/usr/bin/env python3
import argparse
import pickle
import lz4.frame
import numpy as np
import hist
from pathlib import Path

def apply_distortions(h, args):
    """Apply shape distortions to the histogram contents."""
    # Get bin centers and ranges
    axes = h.axes
    if len(axes) != 1:
        # For simplicity, we prioritize the first axis for tilt/curvature
        print(f"Warning: Histogram has {len(axes)} axes. Applying distortions to the first axis only.")
    
    ax = axes[0]
    x = ax.centers
    x_min, x_max = ax.edges[0], ax.edges[-1]
    x_range = x_max - x_min
    x_mean = (x_min + x_max) / 2.0
    
    # Start with original values
    values = h.values(flow=True).copy()
    # Mask for non-flow bins to apply distortions
    mask = np.zeros_like(values, dtype=bool)
    mask[1:-1] = True # simplified for 1D
    
    # 1. Scale
    if args.scale != 1.0:
        values *= args.scale
        print(f"  Applied scale: {args.scale}")
        
    # 2. Tilt: (1 + alpha * (x - x_mean) / x_range)
    if args.tilt != 0.0:
        tilt_factor = 1.0 + args.tilt * (x - x_mean) / x_range
        # Reshape for broadcasting: (len(x), 1, 1...) 
        # Note: values[1:-1] has shape (len(x), M+2, ...)
        dims_to_add = len(values.shape) - 1
        broadcast_tilt = tilt_factor.reshape((len(tilt_factor),) + (1,) * dims_to_add)
        values[1:-1] *= broadcast_tilt
        print(f"  Applied tilt: {args.tilt}")
        
    # 3. Curvature: (1 + beta * ((x - x_mean) / x_range)**2)
    if args.curvature != 0.0:
        curv_factor = 1.0 + args.curvature * ((x - x_mean) / x_range)**2
        dims_to_add = len(values.shape) - 1
        broadcast_curv = curv_factor.reshape((len(curv_factor),) + (1,) * dims_to_add)
        values[1:-1] *= broadcast_curv
        print(f"  Applied curvature: {args.curvature}")
        
    # 4. Localized Bump: N * exp(-(x - mu)**2 / (2 * sigma**2))
    if args.bump_height != 0.0:
        bump = args.bump_height * np.exp(-((x - args.bump_center)**2) / (2 * args.bump_width**2))
        dims_to_add = len(values.shape) - 1
        broadcast_bump = bump.reshape((len(bump),) + (1,) * dims_to_add)
        values[1:-1] += broadcast_bump
        print(f"  Applied bump: height={args.bump_height}, center={args.bump_center}, width={args.bump_width}")
        
    # Ensure no negative values
    values = np.maximum(values, 0.0)
    
    # Create new histogram with distorted values
    new_h = h.copy()
    new_h.view(flow=True).value = values
    # Variances are now just the values (Poisson-like for the "truth")
    new_h.view(flow=True).variance = values
    
    return new_h

def poisson_sample(h):
    """Apply Poisson fluctuations to the histogram."""
    values = h.values(flow=True)
    sampled_values = np.random.poisson(values).astype(float)
    
    new_h = h.copy()
    new_h.view(flow=True).value = sampled_values
    new_h.view(flow=True).variance = sampled_values # Stat uncertainty is sqrt(N)
    return new_h

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-data from a smoothed MC histogram.")
    parser.add_argument("input", type=Path, help="Input .pklz4 file containing the baseline histogram.")
    parser.add_argument("output", type=Path, help="Output .pklz4 file for saving pseudo-data.")
    parser.add_argument("--scale", type=float, default=1.0, help="Overall yield scale factor.")
    parser.add_argument("--tilt", type=float, default=0.0, help="Linear tilt factor (alpha).")
    parser.add_argument("--curvature", type=float, default=0.0, help="Quadratic curvature factor (beta).")
    parser.add_argument("--bump-height", type=float, default=0.0, help="Height of localized Gaussian bump.")
    parser.add_argument("--bump-center", type=float, default=500.0, help="Center of localized Gaussian bump.")
    parser.add_argument("--bump-width", type=float, default=50.0, help="Width (sigma) of localized Gaussian bump.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Poisson sampling.")
    parser.add_argument("--no-poisson", action="store_true", help="Skip Poisson sampling (save distorted truth instead).")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return

    print(f"Loading baseline from {args.input}...")
    with lz4.frame.open(args.input, "rb") as f:
        data = pickle.load(f)
        
    if isinstance(data, dict) and "item" in data:
        h = data["item"]
        metadata = data.get("metadata", {})
    else:
        h = data
        metadata = {}
        
    if not isinstance(h, hist.Hist):
        print(f"Error: Loaded object is not a boost-histogram/hist object. Type: {type(h)}")
        return

    print("Applying distortions...")
    distorted_h = apply_distortions(h, args)
    
    if not args.no_poisson:
        print("Applying Poisson sampling...")
        final_h = poisson_sample(distorted_h)
    else:
        final_h = distorted_h

    # Pack and save
    print(f"Saving pseudo-data to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Record distortions in metadata
    pseudo_metadata = metadata.copy()
    pseudo_metadata["pseudo_data_gen"] = {
        "baseline": str(args.input),
        "scale": args.scale,
        "tilt": args.tilt,
        "curvature": args.curvature,
        "bump": {"height": args.bump_height, "center": args.bump_center, "width": args.bump_width},
        "poisson": not args.no_poisson,
        "seed": args.seed
    }
    
    with lz4.frame.open(args.output, "wb") as f:
        pickle.dump({"item": final_h, "metadata": pseudo_metadata}, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
