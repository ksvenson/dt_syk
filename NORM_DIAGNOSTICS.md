# Degeneracies, moment errors, and plateau interpretation

For Delta_k=rho_T^(k)-rho_RP^(k), partial trace gives Delta_(k-1).
Thus the unnormalized trace norm satisfies ||Delta_1||_1 <= ||Delta_k||_1.
The same conclusion holds if each trace norm is divided by two.

Frobenius norm is not contractive under partial trace. For a single-copy
Hilbert space of dimension D, the general bound is

    ||Delta_k||_HS >= D^(-(k-1)/2) ||Delta_1||_HS.

For the global temporal ensemble, D is the retained sector dimension, not DA.
Consequently the higher-order Frobenius plateaus can be much smaller and can
become visible much later. This does not violate the trace-norm hierarchy.
The code uses absolute Frobenius distances for all these plots.

To see the scaling, take D equally populated eigenvectors in exactly degenerate
pairs, with no further energy-sum resonances. The squared plateau is
P_RP,k(q)-P_RP,k(p), where p has D entries 1/D and q has D/2 entries 2/D.
At large D the floors behave as D^(-1/2), sqrt(6)/D, sqrt(42)/D^(3/2) for
k=1,2,3. Transient decay can therefore conceal the higher floors for a much
longer window. A plateau at k=1 does not force the other curves to flatten at
the same time, even when using trace norm.

An independent NumPy construction of the original pure SYK4 Hamiltonian,
base seed 628^3, realization zero, even parity, gives:

| N | Sector dimension | Detected doublets | Degeneracy-only HS floors k=1,2,3 |
|---:|---:|---:|---|
| 12 | 32 | 16 | 0.03673, 0.02012, 0.01137 |
| 20 | 512 | 256 | 0.02835, 0.004156, 0.0006450 |

These values depend on the eigenvector-phase RP reference chosen by the
numerical eigensolver. They are not predictions of exact saturation times.
Higher-order equal sums beyond doublet degeneracies can increase the floors.
No user plot data was supplied, so this diagnosis establishes consistency,
not a fit or a verification of a particular plotted plateau.

## What is checked and saved

Per-realization JSON now records detected energy blocks, multiplicities,
populations, numerical grouping tolerance, RP purities, degeneracy-only floor
estimates, and the maximum observed violation of the dimension-dependent
Frobenius bound. The default grouping tolerance is 64*machine_epsilon times
the larger of one, spectral width, and maximum absolute energy. It can be
changed with --degeneracy-tolerance, including zero for literal equality.
Grouping is an assumption about unresolved energies; very weak physical
splittings require a smaller tolerance. Nothing is snapped in the evolution.

Independent small-system tests construct full kth-moment matrices, explicitly
perform their partial traces, verify trace-norm monotonicity, exhibit decreasing
Frobenius floors, and compare overlap integration with those full matrices.
The finite-time overlap/purity identity passes these checks with degeneracies.

## A reference-state correction

There is a distinct subtlety in the former diagonal target. With degeneracies,

    sigma_dephased = sum_b Pi_b |psi0><psi0| Pi_b

retains coherences within each degenerate block. It is generally different
from sum_alpha p_alpha |E_alpha><E_alpha| in an arbitrary numerical eigenbasis.
The conditional Scrooge target now uses the former, physical energy-dephased
state. Each occupied block contributes one normalized ket Pi_b|psi0>/sqrt(q_b)
and its weight q_b, so no full density matrix is needed. This fixes the old
target's dependence on rotations inside an exactly degenerate eigenspace.
The grouping tolerance enters this target and is included in its cache name.

The temporal/RP curve deliberately keeps the existing definition: independent
phases for each supplied numerical eigenvector. It need not converge to zero
when energies are degenerate. This RP reference itself is basis dependent in
a degenerate subspace; rotating eigenvectors changes it even though physical
evolution and sigma_dephased remain invariant. Block-phase randomization would
be a different reference, not a correction to that plotted quantity.

For pure SYK4, N=12,20,28 all have N mod 8=4, so testing those sizes is useful.
At N=28 even parity has dimension 8192: one complex128 dense matrix occupies
1 GiB, before eigenvectors and eigensolver workspace. Start with a small number
of simultaneous realizations rather than assuming the N=24 rank allocation
will fit the same node.
