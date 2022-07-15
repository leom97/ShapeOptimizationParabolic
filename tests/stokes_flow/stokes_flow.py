from dolfin import *
import os

if __name__ == "__main__":
    # Test for PETSc or Tpetra
    if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
        info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
        exit()

    if not has_krylov_solver_preconditioner("amg"):
        info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
             "preconditioner, Hypre or ML.")
        exit()

    if has_krylov_solver_method("minres"):
        krylov_method = "minres"
    elif has_krylov_solver_method("tfqmr"):
        krylov_method = "tfqmr"
    else:
        info("Default linear algebra backend was not compiled with MINRES or TFQMR "
             "Krylov subspace method. Terminating.")
        exit()

    # # Meshes and markers for the mesh
    # mesh = Mesh("/home/leonardo_mutti/PycharmProjects/masters_thesis/tests/stokes_flow/dolfin_fine.xml.gz") # mesh wants absolute paths...
    # sub_domains = MeshFunction("size_t", mesh, "/home/leonardo_mutti/PycharmProjects/masters_thesis/tests/stokes_flow/dolfin_fine_subdomains.xml.gz")
    #
    # # Function spaces
    # P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
    # B = VectorFunctionSpace(mesh, "Bubble", 3)
    # Q = FunctionSpace(mesh, "CG", 1)
    # Mini = (P1 + B) * Q