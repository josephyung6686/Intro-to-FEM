# %%
"""
Notes for JY

pip install scikit-fem[all]
https://scikit-fem.readthedocs.io/en/latest/index.html
Paraview 
Blender

"""

"""
We need `pip install scikit-fem[all]`
"""
r"""Linear elasticity.

This example solves the linear elasticity problem using trilinear elements.

"""
import numpy as np
from skfem import *
from skfem.helpers import ddot, sym_grad, eye, trace
from skfem.models.elasticity import lame_parameters


m = MeshHex().refined(3).with_defaults()
e = ElementVector(ElementHex1())
basis = Basis(m, e, intorder=3)

# calculate Lamé parameters from Young's modulus and Poisson ratio
lam, mu = lame_parameters(1e9, 0.4)


def C(T):
    return 2. * mu * T + lam * eye(trace(T), T.shape[0])


@BilinearForm
def stiffness(u, v, w):
    return ddot(C(sym_grad(u)), sym_grad(v))


K = stiffness.assemble(basis)

u = basis.zeros()
u[basis.get_dofs('right').nodal['u^1']] = 0.5

u = solve(*condense(K, x=u, D=basis.get_dofs({'left', 'right'})))

sf = 1.0
m = m.translated(sf * u[basis.nodal_dofs])


if __name__ == "__main__":
    # Define your custom name here
    my_custom_name = "my_elastic_simulation.vtk"

    # Save the mesh AND the displacement data
    # We use u[basis.nodal_dofs].T to attach the movement to the nodes
    m.save(
        my_custom_name, 
        point_data={'displacement': u[basis.nodal_dofs].T}
    )

    print(f"Success! File saved as: {my_custom_name}")
    
    
#%%

r"""Curved elements.

This example solves the eigenvalue problem

.. math::
   -\Delta u = \lambda u \quad \text{in $\Omega$},
with the boundary condition :math:`u|_{\partial \Omega} = 0` using isoparametric
mapping via biquadratic basis and finite element approximation using fifth-order
quadrilaterals.

"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np


p = np.array([[0.  ,  1.  ,  1.  ,  0.  ,  0.5 ,  0.  ,  1.  ,  0.5 ,  0.5 ,
               0.25, -0.1 ,  0.75,  0.9 ,  1.1 ,  0.75,  0.1 ,  0.25,  0.5 ,
               0.25,  0.75,  0.5 ,  0.25,  0.75,  0.75,  0.25],
              [0.  ,  0.  ,  1.  ,  1.  ,  0.  ,  0.5 ,  0.5 ,  1.  ,  0.5 ,
               0.1 ,  0.25, -0.1 ,  0.25,  0.75,  0.9 ,  0.75,  1.1 ,  0.25,
               0.5 ,  0.5 ,  0.75,  0.25,  0.25,  0.75,  0.75]])

t = np.array([[ 0,  4,  8,  5],
              [ 4,  1,  6,  8],
              [ 8,  6,  2,  7],
              [ 5,  8,  7,  3],
              [ 9, 11, 19, 18],
              [17, 12, 13, 20],
              [18, 19, 14, 16],
              [10, 17, 20, 15],
              [21, 22, 23, 24]])

m = MeshQuad2(p, t)
e = ElementQuadP(5)

# create mapping for the finite element approximation and assemble
basis = Basis(m, e)

A = asm(laplace, basis)
M = asm(mass, basis)

L, x = solve(*condense(A, M, D=basis.get_dofs()), solver=solver_eigen_scipy_sym(k=8))

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv
    name = splitext(argv[0])[0]

    from skfem.visuals.matplotlib import *
    ax = draw(m)
    plot(basis, x[:, 6], Nrefs=6, ax=ax)
    savefig(f'{name}_eigenmode.png')
    
    
#%%

r"""Structural vibration.

This example demonstrates the solution of a three-dimensional
vector-valued problem. For this purpose, we consider an elastic
eigenvalue problem.

The governing equation for the displacement of the elastic structure
:math:`\Omega` reads: find :math:`\boldsymbol{u} : \Omega \rightarrow
\mathbb{R}^3` satisfying

.. math::
   \rho \ddot{\boldsymbol{u}} = \mathrm{div}\,\boldsymbol{\sigma}(\boldsymbol{u}) + \rho \boldsymbol{g},
where :math:`\rho = 8050\,\frac{\mathrm{kg}}{\mathrm{m}^3}` is the
density, :math:`\boldsymbol{g}` is the gravitational acceleration and
:math:`\boldsymbol{\sigma}` is the linear elastic stress tensor
defined via

.. math::
   \begin{aligned}
   \boldsymbol{\sigma}(\boldsymbol{w}) &= 2 \mu \boldsymbol{\epsilon}(\boldsymbol{w}) + \lambda \mathrm{tr}\,\boldsymbol{\epsilon}(\boldsymbol{w}) \boldsymbol{I}, \\
   \boldsymbol{\epsilon}(\boldsymbol{w}) &= \frac12( \nabla \boldsymbol{w} + \nabla \boldsymbol{w}^T).
   \end{aligned}
Moreover, the Lamé parameters are given by

.. math::
   \lambda = \frac{E}{2(1 + \nu)}, \quad \mu = \frac{E \nu}{(1+ \nu)(1 - 2 \nu)},
where the Young's modulus :math:`E=200\cdot 10^9\,\text{Pa}`
and the Poisson ratio :math:`\nu = 0.3`.

We consider two kinds of boundary conditions. On a *fixed part* of the boundary, :math:`\Gamma_D \subset \partial \Omega`, the displacement field :math:`\boldsymbol{u}` satisfies

.. math::
   \boldsymbol{u}|_{\Gamma_D} = \boldsymbol{0}.
Moreover, on a *free part* of the boundary, :math:`\Gamma_N = \partial \Omega \setminus \Gamma_D`, the *traction vector* :math:`\boldsymbol{\sigma}(\boldsymbol{u})\boldsymbol{n}` satisfies

.. math::
   \boldsymbol{\sigma}(\boldsymbol{u})\boldsymbol{n} \cdot \boldsymbol{n}|_{\Gamma_N} = 0,
where :math:`\boldsymbol{n}` denotes the outward normal.

Neglecting the gravitational acceleration :math:`\boldsymbol{g}` and
assuming a periodic solution of the form

.. math::
   \boldsymbol{u}(\boldsymbol{x},t) = \boldsymbol{w}(\boldsymbol{x}) \sin \omega t,
leads to the following eigenvalue problem with :math:`\boldsymbol{w}` and :math:`\omega` as unknowns:

.. math::
   \mathrm{div}\,\boldsymbol{\sigma}(\boldsymbol{w}) = \rho \omega^2 \boldsymbol{w}.
The weak formulation of the problem reads: find :math:`(\boldsymbol{w},\omega) \in V \times \mathbb{R}` satisfying

.. math::
   (\boldsymbol{\sigma}(\boldsymbol{w}), \boldsymbol{\epsilon}(\boldsymbol{v})) = \rho \omega^2 (\boldsymbol{w}, \boldsymbol{v}) \quad \forall \boldsymbol{v} \in V,
where the variational space :math:`V` is defined as

.. math::
   V = \{ \boldsymbol{w} \in [H^1(\Omega)]^3 : \boldsymbol{w}|_{\Gamma_D} = \boldsymbol{0} \}.
The bilinear form for the problem can be found from
:func:`skfem.models.elasticity.linear_elasticity`.  Moreover, the mesh
for the problem is loaded from an external file *beams.msh*, which is
included in the source code distribution.

"""
from skfem import *
from skfem.models.elasticity import linear_elasticity,\
                                    lame_parameters
import numpy as np

from pathlib import Path

m = MeshTet.load('beams.msh')
e1 = ElementTetP2()
e = ElementVector(e1)

ib = Basis(m, e)

K = asm(linear_elasticity(*lame_parameters(200.0e9, 0.3)), ib)

rho = 8050.0


@BilinearForm
def mass(u, v, w):
    from skfem.helpers import dot
    return dot(rho * u, v)

M = asm(mass, ib)

L, x = solve(
    *condense(K, M, D=ib.get_dofs("fixed")), solver=solver_eigen_scipy_sym()
)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, show
    sf = 10.0
    m.translated(sf * x[ib.nodal_dofs, 0]).draw().show()