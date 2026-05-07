# FEM Eigenmode Solver

This directory contains two standalone scripts for generating random shapes with holes and solving finite element eigenmode problems:

## Scripts

### 1. `random_shape.py`
Generates randomized shapes with holes and triangulates them using Gmsh.

**Usage:**
```bash
python random_shape.py
```

**Features:**
- Generates smooth random outer boundaries using Fourier harmonics
- Creates random inner holes with validation to ensure they're properly contained
- Triangulates the geometry using Gmsh
- Saves mesh to `output/random_shape.msh`
- Optional visualization of the generated mesh

**Customization:**
The `generate_random_shape_mesh()` function accepts parameters for:
- Outer/inner curve properties (radius, harmonics, noise strength)
- Mesh resolution
- Random seeds for reproducibility
- Output file path

### 2. `fem_solver.py`
Loads a mesh file, solves the FEM eigenvalue problem, and plots eigenmodes.

**Usage:**
```bash
# Basic usage - compute 50 modes and plot mode 0
python fem_solver.py output/random_shape.msh

# Compute 100 modes and plot mode 10 with verbose output
python fem_solver.py output/random_shape.msh --n-modes 100 --mode 10 --verbose

# Plot mode 5 without zero contours
python fem_solver.py output/random_shape.msh --mode 5 --no-zero-contours

# Use a diverging colormap (centered around zero)
python fem_solver.py output/random_shape.msh --mode 10 --colormap RdBu

# Use a different colormap
python fem_solver.py output/random_shape.msh --mode 5 --colormap plasma
```

**Arguments:**
- `mesh_file`: Path to the mesh file (.msh) [required]
- `--n-modes`: Number of eigenmodes to compute (default: 50)
- `--mode`: Index of mode to plot (default: 0)
- `--verbose`: Print eigenvalues
- `--no-zero-contours`: Don't show zero contours on the plot
- `--colormap`: Matplotlib colormap to use (default: YlGn)

## Workflow

1. Generate a random shape mesh:
   ```bash
   python random_shape.py
   ```

2. Solve FEM and visualize modes:
   ```bash
   python fem_solver.py output/random_shape.msh --n-modes 50 --mode 10
   ```

## Dependencies

Both scripts require:
- numpy
- matplotlib
- scipy
- gmsh
- meshio
- skfem

## Output

The FEM solver will display a plot showing:
- The eigenmode amplitude using the specified colormap
- For diverging colormaps (RdBu, RdYlBu, PiYG, PRGn, BrBG, PuOr, RdGy, RdYlGn, Spectral, coolwarm, bwr, seismic), the colormap is centered around zero with vmin=-max|u|, vmax=max|u|
- For other colormaps, the absolute value |u| is plotted
- Zero contours (black lines) where the mode crosses zero (unless disabled)
- The eigenvalue in the title

The eigenmodes represent the natural vibration modes of the shape, with lower mode numbers corresponding to lower frequencies.