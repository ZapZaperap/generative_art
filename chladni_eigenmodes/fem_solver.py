"""Solve FEM eigenvalue problems and save the results to HDF5."""

import argparse
import meshio
import numpy as np
from skfem import MeshTri, Basis, ElementTriP1, asm, condense
from skfem.models.poisson import laplace, mass

from fem_utils import meshio_to_points_triangles, hdf5_filename_for_source, save_fem_solution_hdf5


def solve_fem(m, n_modes, verbose=False):
    points, triangles = meshio_to_points_triangles(m)

    points_c = np.ascontiguousarray(points.T)
    triangles_c = np.ascontiguousarray(triangles.T)
    mesh = MeshTri(points_c, triangles_c)

    element = ElementTriP1()
    basis = Basis(mesh, element)

    K = asm(laplace, basis)
    M = asm(mass, basis)

    D = basis.get_dofs().all()
    Kc, Mc, _, I = condense(K, M, D=D)

    from scipy.sparse.linalg import eigsh
    evals, evecs = eigsh(Kc, M=Mc, k=n_modes, sigma=0.0, which="LM")

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    if verbose:
        print("Eigenvalues:")
        print(evals)

    full_evecs = np.zeros((K.shape[0], n_modes))
    full_evecs[I, :] = evecs
    full_evecs = full_evecs.T

    return mesh, I, evals, full_evecs


def main():
    parser = argparse.ArgumentParser(description="Solve FEM eigenvalue problem and save mode data to HDF5.")
    parser.add_argument("mesh_file", help="Path to the mesh file (.msh)")
    parser.add_argument("--n-modes", type=int, default=50, help="Number of eigenmodes to compute (default: 50)")
    parser.add_argument("--verbose", action="store_true", help="Print eigenvalues")
    parser.add_argument("--output-dir", default=None, help="Directory to write the HDF5 output file")

    args = parser.parse_args()

    print(f"Loading mesh from {args.mesh_file}")
    m = meshio.read(args.mesh_file)

    print(f"Solving FEM problem for {args.n_modes} modes...")
    fem_mesh, I, fem_evals, fem_evecs = solve_fem(m, args.n_modes, verbose=args.verbose)

    output_file = hdf5_filename_for_source(args.mesh_file, args.output_dir)
    save_fem_solution_hdf5(output_file, fem_mesh, fem_evals, fem_evecs, source_mesh_file=args.mesh_file)
    print(f"Saved FEM solution to {output_file}")


if __name__ == "__main__":
    main()
