# The three distance comparisons

The production driver computes exactly these three Hilbert-Schmidt/Frobenius
distances. The projected comparisons share the same evolved states and bath
measurement basis; no extra diagonalization or state evolution is needed.

| Comparison | Target | Saved key |
|---|---|---|
| Temporal versus random phase | Adapted energy-block RP moments | `temp_rp_2norm` |
| Projected versus generalized Scrooge | Fixed energy-dephased full input sigma | `proj_scr_2norm` |
| Projected versus single Scrooge | Instantaneous reduced input rho_A(t) | `proj_single_scr_2norm` |

Here rho_AB(t)=|psi(t)><psi(t)| is the full-system state and
rho_A(t)=Tr_B rho_AB(t) is the reduced state on A. The single Scrooge ensemble
compared with the projected ensemble on A uses rho_A(t), not the full matrix.

Let the state at the sampled time be

$$|\psi(t)\rangle=\sum_z|z\rangle_B\otimes|v_z(t)\rangle_A,$$

with $p_z(t)=\langle v_z(t)|v_z(t)\rangle$ and normalized conditional ket
$|\phi_z(t)\rangle=|v_z(t)\rangle/\sqrt{p_z(t)}$ for nonzero outcomes. The projected
moment is

$$M_k(t)=\sum_z p_z(t)(|\phi_z(t)\rangle\langle\phi_z(t)|)^{\otimes k}.$$

All bath outcomes are summed exactly; zero-probability outcomes contribute zero.

## 1. Generalized Scrooge with energy-dephased input

The original comparison is retained, with the degeneracy correction described
in NORM_DIAGNOSTICS.md:

$$\sigma=\sum_b\Pi_b|\psi_0\rangle\langle\psi_0|\Pi_b,\qquad
G_k[\sigma]=\sum_z q_z S_k[\sigma_z/q_z],$$

where $\sigma_z=\langle z|\sigma|z\rangle$, $q_z=\mathrm{Tr}\sigma_z$, and
$S_k$ is the kth Scrooge moment. The code obtains the conditional blocks directly
from a collection of weighted pure kets representing sigma. It never builds the
full D by D sigma. This target is fixed for a given disorder realization.

## 2. A single Scrooge ensemble with input rho_A(t)

The instantaneous single target is

$$\rho_A(t)=\mathrm{Tr}_B|\psi(t)\rangle\langle\psi(t)|
=\sum_z|v_z(t)\rangle\langle v_z(t)|,\qquad S_k[\rho_A(t)].$$

It is one Scrooge ensemble on A, not a mixture indexed by bath outcomes.
`scrooge_moments` forms it at the very same time t as M_k(t). The target changes
with t; it is not constructed from a time-averaged reduced state.

For a positive matrix R with support dimension r, draw a Haar-uniform normalized
ket h in that support and set v=sqrt(R)h. The probability-adjusted Haar formula is

$$S_k[R]=r\,\mathbb E_h\left[\|v\|^2
\left(\frac{|v\rangle\langle v|}{\|v\|^2}\right)^{\otimes k}\right].$$

For a nonunit-trace R, this means Tr(R) times the Scrooge moment of R/Tr(R).
The factor r and norm weight are essential; the code does not average only
normalized v kets, and does not normalize the resulting sample moment by its
sample trace afterward. Using the occupied support instead of the full DA
space is an exact reformulation of the integral with a lower-variance estimator
when R has zero eigenvalues. Eigenvalues no larger than 64*machine_epsilon times
Tr(R) are treated as roundoff zero. Material negative eigenvalues are rejected.

The steps for a batch of times are:

1. Sum v_z v_z^dagger over z to obtain rho_A, a DA by DA matrix.
2. Diagonalize these small matrices and find their occupied supports.
3. For mixed inputs, normalize independent complex Gaussian vectors in those
   supports to obtain Haar samples. Form v in the eigenbasis and transform it
   back into A's computational basis.
4. Use the same weighted symmetric k-copy coordinates as the projected moments
   and average their outer products with factor r/n_scr.
5. Compare each target with M_k(t) before any time average.

The first moment is exactly S_1[rho_A]=rho_A=M_1 mathematically. The code
subtracts the two computed matrices and returns their numerical residual;
it does not force the first-moment distance to zero.
Rank-one inputs are also analytic for every k, including the initial product
state. Only mixed inputs at higher orders require sampling.

Each sampled time uses independent Haar inputs from RNG stream 3. The fixed
energy-dephased target uses stream 1; sampled times use stream 2; disorder uses
stream 0. The real/imaginary draw ordering makes the instantaneous mixed-state
sample stream independent of evolution batch size. Changing the disorder base
seed changes all streams. Increasing --n-scr refines both sampled target types.

Independent samples do not eliminate the positive contribution of target noise
to the average of a norm. An apparent small plateau should be checked by
increasing n_scr: averaging more times alone does not remove this contribution.
The plot bands remain disorder SEM and do not separately estimate Haar error.

### Parity can obstruct convergence to a single Scrooge ensemble

With fixed full-system fermion parity and computational-basis measurement of B,
every conditional ket on A has definite parity. If rho_A has weight in both
parity sectors, a generic pure ket from single Scrooge(rho_A) can superpose them.
Their first moments agree, but their higher moments need not. A nonzero distance
to this single target can therefore express a conservation constraint rather
than failure of numerical convergence or of the generalized Scrooge description.

For example, the Bell ket (|00>+|11>)/sqrt(2) gives rho_A=I/2 and projected states
|0>,|1> with equal probabilities. Its second projected moment is the equal
mixture of |00><00| and |11><11|, while Scrooge(I/2) is Haar and has second moment
P_sym/3. Their Hilbert-Schmidt distance is exactly 1/sqrt(6), despite equal first
moments. A regression test checks this example against the sampler.

## The former fourth comparison

An earlier version also compared M_k(t) with generalized Scrooge using the
full pure rho_AB(t) as input. Its conditional density matrices are rank one,
so that generalized target equals M_k(t) at every order. It vanishes for all
k, unlike the single Scrooge comparison whose k=1 distance alone is guaranteed
to vanish for an arbitrary state. This fourth comparison, its calculation,
result arrays, MPI collection, and plot have been removed from the driver.

## Averaging, arrays and plots

The temporal/RP distance compares continuously time-averaged global moments.
The two projected distances instead compare each sampled state's projected
moment with its target before averaging those distances over time. Both use
the same stratified running average, or explicit grid averaging if selected.
The single Scrooge input remains rho_A(t) at the sampled time; it is never
replaced by the reduced time-averaged state.

The three saved distance arrays have shape (realizations,times,k), and each
has its own threshold-time array. They are collected across MPI ranks and
saved with NPZ compression. No first-moment single-Scrooge distance is replaced
with zero. Tests perturb that target to verify that a discrepancy survives
numerical comparison, time averaging, MPI collection, saving, and plotting.

New result and projected-intermediate filenames include `singlecheck` to
avoid reusing intermediates with the old analytically filled first-moment
column. Metadata identifies `single_scrooge_evaluation` as
`numerical moment difference`. Existing archives may contain an extra fourth
array; the sweep plot reads only the three comparisons listed above.

A production run generates three figures. `plot_N_g_sweep.py --results ...`
generates those three across N/g plus the original temporal-versus-fixed-target
parametric figure. The single-Scrooge k=1 sweep panel has a linear vertical
axis with an automatic upper limit, so a nonzero numerical error is visible.
Other moment orders of this comparison can be physically nonzero and sampled.
