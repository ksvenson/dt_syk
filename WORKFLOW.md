# Computational workflow and independent checks

This describes the production calculation in `syk_temp_proj.py` on the
`optimize-temporal-sampling` branch. Its two final curves have different
averaging procedures, specified below. Energies and times use units with
hbar = 1.

## 1. Draw and diagonalize the Hamiltonian

For each disorder realization, construct

$$H=H_4+\frac{g}{N}H_2.$$

Here N counts Majorana modes and L=N/2 counts two-level sites. The full
Hilbert-space dimension is D=2^L. With the current vacuum initial state
$|\psi_0\rangle=|0\rangle$ and even-body interactions, dynamics remain in even
fermion parity. The default calculation therefore diagonalizes the D/2 by
D/2 even-parity block:

$$H|E_\alpha\rangle=E_\alpha|E_\alpha\rangle,\qquad
c_\alpha=\langle E_\alpha|\psi_0\rangle,\qquad
p_\alpha=|c_\alpha|^2.$$

`evecs[:,alpha]` is the energy eigenket; `evals[alpha]` is its energy.
`psi0_eng_basis[alpha]` is c_alpha; `pops[alpha]` is p_alpha.

**Checks:** $\sum_\alpha p_\alpha=1$, orthonormal eigenvectors, and agreement
between the sector Hamiltonian and the corresponding block of the full
Hamiltonian using identical disorder couplings. Full computational-basis
labels are retained explicitly; sector row numbers cannot be interpreted as
subsystem labels directly. Restriction to even parity relies on this initial
state and Hamiltonian and is not a prescription for arbitrary initial states.

## 2. Compute the random-phase reference exactly

The independent random-phase states are

$$|\psi_{\boldsymbol\theta}\rangle
=\sum_\alpha c_\alpha e^{i\theta_\alpha}|E_\alpha\rangle,$$

with each phase uniform on [0,2 pi). The kth moment is the phase average of
$(|\psi_{\boldsymbol\theta}\rangle\langle\psi_{\boldsymbol\theta}|)^{\otimes k}$.
We need its purity, not its full matrix:

$$P_{\mathrm{RP},k}=\operatorname{Tr}[(\rho_{\mathrm{RP}}^{(k)})^2]
=\sum_{\substack{n_\alpha\ge0\\\sum_\alpha n_\alpha=k}}
\left(\frac{k!}{\prod_\alpha n_\alpha!}\right)^2
\prod_\alpha p_\alpha^{2n_\alpha}.$$

`random_phase.rpe_square_2norm(pops,k_max)` computes all purities from order
1 through k_max using coefficients of a polynomial expansion. It does not
sample phases and does not assume that energy sums are nondegenerate.

**Checks:** define $s_j=\sum_\alpha p_\alpha^j$. Then

$$P_{\mathrm{RP},1}=s_2,\qquad
P_{\mathrm{RP},2}=2s_2^2-s_4,\qquad
P_{\mathrm{RP},3}=6s_2^3-9s_2s_4+4s_6.$$

A state occupying one energy eigenket has every RP purity equal to one.
For small dimensions the result can also be checked by constructing the
full phase-averaged moment with `rpe_moment` and calculating its purity.

## 3. Compute continuous temporal moments from scalar overlaps

For a uniformly drawn time in [0,tau], the temporal moment is

$$\rho_T^{(k)}(\tau)=\frac1\tau\int_0^\tau
(|\psi(t)\rangle\langle\psi(t)|)^{\otimes k}\,dt,$$

where $|\psi(t)\rangle=\sum_\alpha c_\alpha e^{-iE_\alpha t}|E_\alpha\rangle$.
Stationarity gives the return amplitude and purity formula

$$A(u)=\langle\psi_0|\psi(u)\rangle=\sum_\alpha p_\alpha e^{-iE_\alpha u},$$

$$P_{T,k}(\tau)=\frac{2}{\tau^2}\int_0^\tau
(\tau-u)|A(u)|^{2k}\,du.$$

The overlap with the RP moment obeys
$\operatorname{Tr}[\rho_T^{(k)}\rho_{\mathrm{RP}}^{(k)}]=P_{\mathrm{RP},k}$.
Consequently the first curve is

$$d_{T,k}(\tau)=\|\rho_T^{(k)}(\tau)-\rho_{\mathrm{RP}}^{(k)}\|_{\mathrm{HS}}
=\sqrt{P_{T,k}(\tau)-P_{\mathrm{RP},k}}.$$

Here the Hilbert-Schmidt norm means
$\|X\|_{\mathrm{HS}}=\sqrt{\operatorname{Tr}(X^\dagger X)}$.
The returned key `'2norm'` elsewhere in the repository uses this same norm;
it is not the trace norm or the largest singular value.

`temporal_distance_continuous` evaluates spectral overlaps on a fine grid and
integrates $f_k(u)=|A(u)|^{2k}$ and $u f_k(u)$ by Simpson's rule. It then
combines their integrals to get every requested tau. It never needs the
large k-copy temporal moment matrix or the full evolved state. At tau=0,
the initial-state moment is used, giving $d_{T,k}(0)=\sqrt{1-P_{\mathrm{RP},k}}$.

`--dt` is an upper bound on the integration step. It is reduced as necessary
to fit an even number of subdivisions between consecutive reporting times.
The default `auto` chooses at most pi/(2 k_max W), where W is the width of
the computed spectrum, and at most 0.1. `spectral_sampling_diagnostics`
reports the conservative frequency-resolution condition dt < pi/(k_max W).
Resolving the frequencies does not itself specify an integration accuracy.

**Checks:** compare dt and dt/2 with `--check-dt`. For a small spectrum, compare
with explicit sums over ordered k-tuples alpha and beta:

$$P_{T,k}(\tau)=\sum_{\boldsymbol\alpha,\boldsymbol\beta}
\left(\prod_{j=1}^k p_{\alpha_j}p_{\beta_j}\right)
\operatorname{sinc}^2\!\left[
\frac{\tau}{2}\sum_{j=1}^k(E_{\alpha_j}-E_{\beta_j})\right],$$

where sinc(x)=sin(x)/x, with sinc(0)=1. NumPy uses a pi-scaled definition;
the tests therefore pass the displayed argument divided by pi to `np.sinc`.
Repeating an energy index in a tuple is allowed. Shifting all energies by
one constant must leave the distance unchanged. Resonances can leave a
nonzero limiting distance; the code does not assume thermalization occurs.

## 4. Build the conditional Scrooge target

First define the energy-dephased density matrix

$$\sigma=\sum_b\Pi_b|\psi_0\rangle\langle\psi_0|\Pi_b,$$

where Pi_b projects onto a full degenerate energy subspace. The calculation
retains coherences inside each block. Without degeneracies this reduces to
sum_alpha p_alpha |E_alpha><E_alpha|. See NORM_DIAGNOSTICS.md for the numerical
energy grouping tolerance and its effect on the independent eigenvector-phase
RP reference, which remains the definition of the temporal comparison.

The full basis is ordered as $|z\rangle_B\otimes|a\rangle_A$, with integer
label z*DA+a, so a full ket reshapes to `(DB,DA)`. The default NA=4 Majoranas
gives DA=4 and DB=D/DA. Measuring B in this computational basis defines

$$\sigma_z=\langle z|\sigma|z\rangle,\quad
q_z=\operatorname{Tr}\sigma_z,\quad
\widehat\sigma_z=\sigma_z/q_z\quad(q_z>0).$$

`conditional_covariances` returns the **unnormalized** matrices sigma_z
by summing eigenvector amplitudes directly. The name refers to these density
matrices, not to a classical covariance of random variables.

The target is a bath-weighted mixture of conditional Scrooge moments:

$$C_k=\sum_z q_z\,S_k(\widehat\sigma_z).$$

For Haar-uniform normalized kets h in DA dimensions, let
$v_z=\sqrt{\sigma_z}\,h$ and $w_z=\langle v_z|v_z\rangle$. The Scrooge definition
can then be evaluated as

$$C_k=D_A\sum_z\int d\mu(h)\,w_z
\left(\frac{|v_z\rangle\langle v_z|}{w_z}\right)^{\otimes k}.$$

`cond_scr_moment` approximates this integral using n_scr independent Haar
input kets, obtained by normalizing independent complex Gaussian vectors.
It sums all bath outcomes. The same Haar inputs are reused for every z,
which gives the correct expectation but correlated errors across outcomes.
The extra weight w_z is essential. Bath probabilities q_z are already
included in sigma_z and must not be multiplied in again.

For k=1 the code uses the exact identity

$$C_1=\sum_z\sigma_z=\operatorname{Tr}_B\sigma.$$

For k>1 the Haar integral is sampled. Its estimated trace fluctuates around
one; the code does not renormalize it afterward. Zero-weight outcomes make
zero contributions, and rank-deficient matrices are allowed.

**Checks:** sum_z q_z=1; positive semidefinite sigma_z; C_1 equals the reduced
sigma; convergence when increasing n_scr or changing the Haar random seed.
For a pure conditional density matrix, the corresponding weighted Scrooge
moment must agree in expectation with that same pure-state moment. The tests
also compare the sampled formula in the full and symmetric tensor bases
using exactly the same Haar kets.

## 5. Evolve states, measure B, and compare their projected moments

At a sampled time t, expand the sector ket into the full basis, then write

$$|\psi(t)\rangle=\sum_z|z\rangle_B\otimes|v_z(t)\rangle_A.$$

The Born probability is $p_z(t)=\langle v_z(t)|v_z(t)\rangle$. With
$|\phi_z(t)\rangle=|v_z(t)\rangle/\sqrt{p_z(t)}$, the projected moment is

$$M_k(t)=\sum_z p_z(t)
(|\phi_z(t)\rangle\langle\phi_z(t)|)^{\otimes k}.$$

`projected_moment` evaluates this sum exactly over bath outcomes. To do so
it uses `weighted_features`, the amplitudes of
$\sqrt{p_z}|\phi_z\rangle^{\otimes k}$ in an orthonormal symmetric basis.
`symmetric_basis` labels that basis by occupation tuples and supplies their
normalization factors. `symmetric_isometry` converts it to the full tensor
basis for verification. The smaller representation changes neither the
operator nor its Hilbert-Schmidt norm.

`time_evolved_overlaps_and_ps_2norm` returns, for each sampled time,

$$d_{P,k}(t)=\|M_k(t)-C_k\|_{\mathrm{HS}},$$

as well as the complex return amplitude. It does not take a time average.
The global phase removed during state reconstruction has no effect on M_k.

**Checks:** Tr(M_k(t))=1, positive semidefinite M_k(t),
$M_1(t)=\operatorname{Tr}_B|\psi(t)\rangle\langle\psi(t)|$, finite values at
zero time, and equal distances in the full and symmetric bases. On a small
system, full-space and even-sector evolution should give the same projected
moments after the correct basis correspondence is applied.

## 6. Average projected distances over time and collect disorder statistics

The second curve estimates

$$\overline d_{P,k}(\tau)=\frac1\tau\int_0^\tau d_{P,k}(t)\,dt.$$

For reporting interval width Delta, the default samples one independent
uniform time t_j in each interval [(j-1)Delta,j Delta). At tau=j Delta,
the estimate is the mean of d_P,k(t_1),...,d_P,k(t_j). The zero-time distance
is stored separately and is not added to these continuous-time estimates.
The reporting width is adjusted slightly if needed to end exactly at tmax.

This time average of distances generally differs from

$$\left\|\frac1\tau\int_0^\tau M_k(t)\,dt-C_k\right\|_{\mathrm{HS}}.$$

The code preserves the original question of how far the projected ensemble
of a state drawn from the temporal ensemble lies from its target. It does
not compute the displayed distance of an averaged projected moment.
`--projected-sampling grid` instead averages distances at regular endpoints,
including zero time; that is a discrete-time observable. Both choices keep
the separate temporal curve as a continuous integration estimate.

`--projected-dt` controls how many full states are sampled, independently of
`--dt`. Draws are reproducible. Changing `--time-seed` varies only the time
sampling; changing `--seed` also changes disorder and Scrooge samples.

Finally, each comparison is averaged over disorder realizations. The code
averages distances, not their squares. Plot shading is the standard error
across disorder realizations; it does not separately quantify Scrooge or
time-sampling uncertainty. Each realization has its own sampled Scrooge target.

**Checks:** repeat projected sampling with a different time-seed and/or a
smaller projected-dt; refine n_scr independently; hold the number of disorder
realizations fixed when changing the number of MPI processes.

## What is exact, and what needs convergence checks?

| Quantity | Treatment | Main independent check |
|---|---|---|
| Eigenkets and spectral phases | Numerical diagonalization and direct phase evaluation | Small full/sector comparison |
| RP purity | Exact finite algebra, up to floating-point error | Brute-force small moments |
| Temporal purity integral | Simpson integration | dt versus dt/2; exact small-spectrum formula |
| Projection at a specified time | Exact sum over all bath outcomes | Direct full-tensor moments |
| Symmetric basis reduction | Exact change of representation | U^dagger U=I and equal distances |
| Conditional Scrooge k=1 | Exact reduced sigma | Sum conditional blocks |
| Conditional Scrooge k>1 | Haar sampling | Increase n_scr; independent seed |
| Projected time average | One random time per interval | Refine projected-dt; vary time-seed |
| Disorder statistics | Finite independent realizations | Increase realization count |

Intermediate calculations saved under a `note` are identified by that name,
not by comparing all their physical inputs. For independent verification with
changed arrays or seeds, choose a fresh note or a fresh cache directory.

The numerical tests in `tests/` implement the small-system comparisons above.
`tests/test_hamiltonian.py` additionally compares the production factory with
an independent Majorana construction when dynamite is installed. See
`PERFORMANCE.md` for runtime measurements and the remaining cluster validation.


## Cache controls and plotted timescale diagnostics

`utilities.CACHE_DIR` is now kept as a static default. The production driver
passes its chosen directory explicitly, checks the eigensystem first, and skips
Hamiltonian construction on a saved-eigensystem hit. Hamiltonian caching and
redundant projected intermediates are opt-in. No existing files are removed.

Optional spectral lines show 2*pi/minimum k-sum gap, not a claimed convergence
time. Repeated indices in a sum are allowed; permutations of the same label
are excluded. Sorting all sums avoids comparing every pair, and an enumeration
cap prevents prohibitive higher-order allocations. Zero and unresolved gaps
are recorded without a finite line. An optional observed relative-distance
threshold gives a direct comparison of the curves within their sampled window.

See PARAMETERS.md for the precise definitions, all controls and defaults, storage
estimates, and examples. `thermalization.py` documents the calculations.
