from mpi4py import MPI

comm = MPI.COMM_WORLD
if comm.rank == 0:
    print("MPI library version:")
    print(MPI.Get_library_version())
