import dynamite as dt
import dynamite.operators as op


def MFIM(hx=0.9045, hz=0.8090, J=1, boundary='open'):
    """
    Mixed field Ising model.
    L should be set with `dynamite.config.L`

    Default values come from  10.1103/PhysRevE.90.052105.
    """
    # separate field and interaction terms so that boundary conditions can be applied
    # properly.
    field = op.index_sum(hx * op.sigmax() + hz * op.sigmaz())
    interaction = op.index_sum(J * op.sigmaz(i=0) * op.sigmaz(i=1), boundary=boundary)
    return field + interaction

