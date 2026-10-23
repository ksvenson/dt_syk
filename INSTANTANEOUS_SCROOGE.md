# The three projected-ensemble targets

The production run now calculates the original energy-dephased target and the
two requested instantaneous targets from the **same evolved states and bath
measurement basis**. It does not repeat diagonalization or full-state evolution
for the new comparisons. All distances are Hilbert-Schmidt/Frobenius distances.

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

The first new target is

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

The first moment is exactly S_1[rho_A]=rho_A=M_1, so that distance is exactly zero.
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

## 3. Generalized Scrooge with the full instantaneous input rho(t)

For the second new target, the full input is **pure**:

$$\rho(t)=|\psi(t)\rangle\langle\psi(t)|.$$

Its bath-conditional matrices are

$$\rho_z(t)=\langle z|\rho(t)|z\rangle
=|v_z(t)\rangle\langle v_z(t)|.$$

Each nonzero conditional matrix has rank one. The Scrooge ensemble of a pure
density matrix contains just that pure ket, so

$$G_k[\rho(t)]=\sum_z p_z(t)
S_k[|\phi_z(t)\rangle\langle\phi_z(t)|]=M_k(t).$$

Thus this comparison is **identically zero for all t and k**, independently of
g, N, chaos, or thermalization. This conclusion uses the generalized-Scrooge
mixture defined by the same bath measurement basis. A different mixed global
input or different bath basis would be a different calculation.

The code implements the identity analytically. There is no D by D pure density
matrix allocation, no per-outcome square root, and no sampling needed. The
`generalized_scrooge_pure` function supplies the moment itself for independent
small-system verification. Tests also construct the rank-one conditional blocks
and apply the separately implemented Scrooge formula before summing them.
The driver returns exact zeros without evaluating an identical moment twice.
This plot is a consistency identity, not a thermalization diagnostic.

## Averaging, arrays and plots

At every sampled time the three distances are calculated against their own
specified targets. The positive-time curves then use the **same** stratified
running average (or explicit grid averaging if selected) as the original
projected comparison. They are averages of distances, not distances between
averaged moments. The target of the first new comparison remains rho_A(t)
inside that average; it is never replaced by the reduced time-averaged state.

The production NPZ now contains, each with shape (realizations,times,k):

| Key | Per-state quantity averaged in time |
|---|---|
| `proj_scr_2norm` | distance to fixed G_k[sigma_dephased] |
| `proj_single_scr_2norm` | distance to S_k[rho_A(t)] |
| `proj_instantaneous_gse_2norm` | distance to G_k[rho(t)]; exactly zero |

`temp_rp_2norm` retains its separate continuous temporal/RP meaning. The files
also contain separate observed threshold-time arrays for the new curves.
Archives use NPZ compression, which makes the analytically zero array cheap
on disk. That array is broadcast from a scalar before saving, so it also does
not require an additional full result cube in RAM or an MPI collection.

A production run generates four figures; `plot_N_g_sweep.py --results ...`
generates those four comparisons across N/g plus the original temporal-versus-
fixed-target parametric plot. Exact-zero curves use a linear vertical axis or
are labeled as exactly zero when other orders need a logarithmic axis. The
per-realization projected stage duration includes constructing instantaneous
single-Scrooge targets; these add work and require a new cluster timing pilot.
