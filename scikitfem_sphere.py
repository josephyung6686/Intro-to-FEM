# %%

import numpy as np
import gmsh
from skfem import *
from skfem.helpers import sym_grad, eye, trace, dot, inner
from skfem.models.elasticity import lame_parameters


# ==========================================
# 1. MESH GENERATION (QUADRATIC)
# ==========================================
def create_uniform_hollow_sphere(outer_radius, thickness, mesh_size, output_name="hollow_sphere"):
    """Generates a highly uniform, Quadratic (Second-Order) 3D mesh."""
    inner_radius = outer_radius - thickness
    if inner_radius <= 0:
        raise ValueError("Thickness must be strictly less than the outer radius.")

    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", 0)
    gmsh.model.add("hollow_sphere")

    # Geometry Construction
    outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, outer_radius)
    inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, inner_radius)
    gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)])
    gmsh.model.occ.synchronize()

    # Apply the Controllable Mesh Size
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # Disable adaptive sizing to force uniformity
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Use HXT (10) 3D algorithm for fast, high-quality uniform tets
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)

    print(f"Generating uniform 3D mesh (Size: {mesh_size})...")
    gmsh.model.mesh.generate(3)

    # --- UPGRADE: Convert to Quadratic (10-Node) Tetrahedrons ---
    print("Converting elements to Quadratic (Second-Order) curvature...")
    gmsh.model.mesh.setOrder(2)

    msh_file = f"{output_name}.msh"
    gmsh.write(msh_file)
    print(f"Mesh generated successfully: {msh_file}")

    gmsh.finalize()
    return msh_file


# ==========================================
# 2. FEA SOLVER (QUADRATIC BASIS)
# ==========================================
def solve_elasticity_and_stress(msh_file, r_out, r_in, p_out, p_in, E=200e9, nu=0.3):
    print("Loading quadratic mesh and assembling system...")
    mesh = Mesh.load(msh_file)
    lambda_, mu = lame_parameters(E, nu)

    # --- UPGRADE: Use P2 (Quadratic) Elements ---
    elem = ElementVector(ElementTetP2())
    basis = Basis(mesh, elem)

    @BilinearForm
    def linear_elasticity(u, v, w):
        def C(T):
            return 2 * mu * T + lambda_ * eye(trace(T), 3)
        return inner(C(sym_grad(u)), sym_grad(v))

    A = asm(linear_elasticity, basis)

    @LinearForm
    def pressure_inner(v, w): return dot(-p_in * w.n, v)

    @LinearForm
    def pressure_outer(v, w): return dot(-p_out * w.n, v)

    # Find boundaries
    inner_facets = mesh.facets_satisfying(lambda x: np.linalg.norm(x, axis=0) < r_in + 0.1)
    outer_facets = mesh.facets_satisfying(lambda x: np.linalg.norm(x, axis=0) > r_out - 0.1)

    b = asm(pressure_inner, FacetBasis(mesh, elem, facets=inner_facets)) + \
        asm(pressure_outer, FacetBasis(mesh, elem, facets=outer_facets))

    # --- Dirichlet BCs: Safe Vertex Targeting ---
    # Quadratic meshes have mid-edge nodes. We only want to search among corner vertices.
    n_vertices = basis.nodal_dofs.shape[1]
    vertices = mesh.p[:, :n_vertices]

    bottom_node = np.argmin(vertices[2])  # Lowest Z
    top_node = np.argmax(vertices[2])     # Highest Z
    side_node = np.argmax(vertices[0])    # Highest X

    D = np.concatenate((
        basis.nodal_dofs[:, bottom_node],        # Fix Bottom: X, Y, Z
        basis.nodal_dofs[0:2, top_node],         # Fix Top: X, Y
        [basis.nodal_dofs[1, side_node]]         # Fix Side: Y
    ))

    # --- Solve Displacement ---
    print("Solving displacement matrix...")
    u = solve(*condense(A, b, D=D))

    # --- Calculate Stress & Quality ---
    print("Calculating von Mises stress and mesh quality...")
    basis0 = Basis(mesh, ElementTetP0(), quadrature=basis.quadrature)

    @LinearForm
    def von_mises_form(v, w):
        eps = sym_grad(w.u)
        sig = 2 * mu * eps + lambda_ * eye(trace(eps), 3)
        sig_dev = sig - eye(trace(sig) / 3.0, 3)
        return np.sqrt(1.5 * inner(sig_dev, sig_dev)) * v

    vm_integral = asm(von_mises_form, basis0, u=basis.interpolate(u))

    @BilinearForm
    def p0_mass(u, v, w): return u * v

    element_volumes = asm(p0_mass, basis0).diagonal()
    von_mises_stress = vm_integral / element_volumes

    # Mesh Quality (Mean Ratio Metric)
    p0 = mesh.p[:, mesh.t[0]]
    p1 = mesh.p[:, mesh.t[1]]
    p2 = mesh.p[:, mesh.t[2]]
    p3 = mesh.p[:, mesh.t[3]]

    sum_edge_lengths_sq = (
        np.sum((p1 - p0)**2, axis=0) + np.sum((p2 - p0)**2, axis=0) +
        np.sum((p3 - p0)**2, axis=0) + np.sum((p2 - p1)**2, axis=0) +
        np.sum((p3 - p1)**2, axis=0) + np.sum((p3 - p2)**2, axis=0)
    )
    mesh_quality = (72.0 * np.sqrt(3.0) * np.abs(element_volumes)) / (sum_edge_lengths_sq**1.5)

    # --- Export ---
    out_file = 'sphere_results_quadratic.vtk'

    # 1. Grab displacements for the corner nodes (shape: 3 x N_nodes)
    u_nodes = u[basis.nodal_dofs]

    # 2. Grab displacements for the mid-edge nodes (shape: 3 x N_edges)
    u_edges = u[basis.edge_dofs]

    # 3. Combine them to match the exact [nodes, edges] order of mesh.p
    u_full = np.hstack((u_nodes, u_edges))

    # 4. Save to VTK
    mesh.save(
        out_file,
        point_data={'u': u_full.T},
        cell_data={
            'von_mises': [von_mises_stress],
            'element_quality': [mesh_quality]
        }
    )

    print(f"Minimum Element Quality: {np.min(mesh_quality):.3f} (closer to 1.0 is better)")
    print(f"Maximum von Mises Stress: {np.max(von_mises_stress) / 1e6:.2f} MPa")
    print(f"Complete! Results saved to {out_file}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
# --- Geometry & Mesh Controls ---
R_OUTER = 10.0
THICKNESS = 2.0
R_INNER = R_OUTER - THICKNESS

# CONTROLLABLE MESH SIZE:
# 2.0 = Coarse (Fastest)
# 1.0 = Medium
# 0.5 = Fine (Beautiful resolution, takes a minute to solve)
TARGET_ELEMENT_SIZE = 1

# Execute Pipeline
mesh_filename = create_uniform_hollow_sphere(
    outer_radius=R_OUTER,
    thickness=THICKNESS,
    mesh_size=TARGET_ELEMENT_SIZE,
    output_name="quadratic_sphere"
)

solve_elasticity_and_stress(
    msh_file=mesh_filename,
    r_out=R_OUTER,
    r_in=R_INNER,
    p_out=0,  # 10 MPa External Pressure
    p_in=50e6    # 50 MPa Internal Pressure
)
