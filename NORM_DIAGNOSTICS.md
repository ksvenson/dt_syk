# Understanding moment errors, degeneracies, and the reference ensembles

A small distance only has meaning after specifying what is being compared.
This calculation uses two different references: a random-phase ensemble for
the global temporal comparison, and a generalized Scrooge ensemble for the
subsystem projected comparison. Degeneracies affect both, but in different ways.

Throughout this note, hbar=1 and D is the dimension of **one copy of the global
Hilbert space used in the calculation**. With even parity, D=2^(N/2-1). In
particular, N=20 gives D=512. This D is not DA, the dimension of subsystem A,
and differs from the full-space dimension denoted D in parts of WORKFLOW.md.

## Current reference: one occupied ket per energy eigenspace

After loading the eigensystem, `initial_state_eigenbasis` rotates the basis
within each degenerate energy eigenspace. Its first ket is

$$
q_b=\|\Pi_b|\psi_0\rangle\|^2,\qquad
|\chi_b\rangle=\frac{\Pi_b|\psi_0\rangle}{\sqrt{q_b}}
\quad(q_b>0).
$$

The remaining kets are orthogonal to the initial state, with zero initial
population. The production temporal/RP comparison now uses this adapted basis.
Equivalently its RP reference is

$$
|\psi_{\boldsymbol\theta}\rangle
=\sum_b\sqrt{q_b}\,e^{i\theta_b}|\chi_b\rangle.
$$

It depends only on the Hamiltonian's energy projectors and the initial ket;
the arbitrary completion of each eigenspace ONB has no effect. Its first moment
is exactly the physical dephased state sigma. Thus the first temporal moment
converges to it, including for a degenerate Hamiltonian. This is an infinite-time
statement, with finite integration error in the numerical calculation. It is
not an assertion that an instantaneous evolved pure state becomes mixed.

The physical ket, density matrix, temporal moments, energy projectors, and
sigma are basis-independent operators. An independent-phase RP prescription
in an arbitrary basis generally depends on that basis. The adapted prescription
above removes dependence on arbitrary choices inside an energy eigenspace.
Higher-order resonances among distinct occupied energies can still obstruct
convergence of higher temporal moments to this reference.

For exact degeneracies the rotation leaves the Hamiltonian and evolution
unchanged. Numerically grouped energies use a common block mean; the effect
of this explicit approximation is described under numerical checks below.

## Moments describe the same ensemble with more copies

An ensemble is a probability distribution of normalized pure kets. Its first
moment is the usual average density matrix. Its kth moment describes preparing
k copies of the same randomly selected ket:

$$
M_k=\mathbb E_{\psi}\left[(|\psi\rangle\langle\psi|)^{\otimes k}\right].
$$

The copies share the same selected ket; they are not independently drawn. This
is why M_k generally differs from the kth tensor power of the average density
matrix. The correlations between copies contain information that the average
state alone does not capture.

Tracing out one copy of a pure-state projector contributes its trace, which is
one. Averaging does not change this identity:

$$
\mathrm{Tr}_{\text{one copy}}M_{k+1}=M_k.
$$

It holds separately for the temporal and random-phase ensembles. Consequently,
if their kth-moment difference is

$$
\Delta_k(\tau)=\rho_T^{(k)}(\tau)-\rho_{\mathrm{RP}}^{(k)},
$$

then tracing out k-r copies gives Delta_r. These partial traces are over
**copies of the entire system**, not over the physical bath B used to construct
projected ensembles.

## Why trace norm and Frobenius norm give different comparisons

The trace norm of a Hermitian operator is the sum of the absolute values of its
eigenvalues. Half the trace norm of a difference of density matrices is the
usual trace distance, which measures their distinguishability. Discarding some
copies cannot improve distinguishability. Thus

$$
\|\Delta_r\|_1\le\|\Delta_k\|_1,\qquad 1\le r\le k.
$$

Your partial-trace argument establishes exactly this statement. The conclusion
is unchanged if every trace norm is divided by two. It does not require the
curves to begin flattening at the same time: it constrains their heights, not
where their transient and asymptotic regimes meet.

The Frobenius norm, also called the Hilbert-Schmidt norm, instead treats matrix
elements like components of a Euclidean vector:

$$
\|X\|_{\mathrm{HS}}^2=\mathrm{Tr}(X^\dagger X)=\sum_{i,j}|X_{ij}|^2.
$$

This norm is not contractive under partial trace. The reason is that a partial
trace **adds** certain matrix elements before their absolute squares are taken.
Those terms can reinforce one another.

To see the precise factor, write an arbitrary operator X on a retained space
and a discarded space of dimension m. If i,j label the retained space and a,b
label the discarded space, then

$$
(\mathrm{Tr}_{\mathrm{discarded}}X)_{ij}=\sum_{a=1}^{m}X_{ia,ja}.
$$

Cauchy-Schwarz says that the squared magnitude of a sum of m terms is at most
m times their sum of squared magnitudes. Applying it to every matrix element,

$$
\begin{aligned}
\|\mathrm{Tr}_{\mathrm{discarded}}X\|_{\mathrm{HS}}^2
&=\sum_{i,j}\left|\sum_a X_{ia,ja}\right|^2\\
&\le m\sum_{i,j,a}|X_{ia,ja}|^2\\
&\le m\sum_{i,j,a,b}|X_{ia,jb}|^2
=m\|X\|_{\mathrm{HS}}^2.
\end{aligned}
$$

The last inequality simply includes the additional nonnegative terms with
unequal discarded indices. Taking square roots gives

$$
\|\mathrm{Tr}_{\mathrm{discarded}}X\|_{\mathrm{HS}}
\le\sqrt m\,\|X\|_{\mathrm{HS}}.
$$

For k-copy moments reduced to r copies, the discarded space has dimension
m=D^(k-r). Substituting X=Delta_k therefore gives

$$
\|\Delta_k\|_{\mathrm{HS}}
\ge D^{-(k-r)/2}\|\Delta_r\|_{\mathrm{HS}}.
$$

In particular,

$$
\|\Delta_k\|_{\mathrm{HS}}
\ge D^{-(k-1)/2}\|\Delta_1\|_{\mathrm{HS}}.
$$

For N=20 in even parity, the factors relating the second and third errors to
the first are 1/sqrt(512) and 1/512. A finite first-order error is compatible
with much smaller higher-order Frobenius errors. The inequality does not
predict a nonzero asymptotic floor for the current adapted reference.

The dimension factor is real, not just an artifact of a loose proof. For any
operator Y, take X=Y tensor I_m/m. Its partial trace is Y, whereas

$$
\|X\|_{\mathrm{HS}}=\frac{\|Y\|_{\mathrm{HS}}}{\sqrt m}.
$$

This example concerns general operators; it illustrates the norm property,
without asserting that every such X is a moment difference in the SYK run.
The bound need not be attained by the particular moment differences being
studied, so it alone does not predict their plateau values.

## What the random-phase reference actually is

Choose an orthonormal energy eigenbasis and expand the initial ket:

$$
|\psi_0\rangle=\sum_\alpha c_\alpha|E_\alpha\rangle,
\qquad p_\alpha=|c_\alpha|^2.
$$

Actual unitary evolution produces

$$
|\psi(t)\rangle=\sum_\alpha c_\alpha e^{-iE_\alpha t}|E_\alpha\rangle.
$$

The random-phase, or RP, reference is a separate mathematical ensemble. Keep
these amplitudes and this basis fixed, but replace the dynamical phases with
independent uniform phases:

$$
|\psi_{\boldsymbol\theta}\rangle
=\sum_\alpha c_\alpha e^{i\theta_\alpha}|E_\alpha\rangle,
\qquad \theta_\alpha\ \text{independent and uniform on }[0,2\pi).
$$

Its kth moment is the average of the k-copy projector over all those phases.
Thus the reference is an **ensemble**, not a single randomly chosen ket. At
first order the independent phase average removes every term with unequal
basis indices:

$$
\rho_{\mathrm{RP}}^{(1)}
=\sum_\alpha p_\alpha|E_\alpha\rangle\langle E_\alpha|.
$$

This is the general eigenvector-phase definition. In current production runs,
the eigenbasis is first adapted as described above, so only one coefficient
per energy block is nonzero. The temporal comparison measures distance to the
moments of that adapted ensemble. The code evaluates their purities exactly;
it does not estimate them by sampling phases.

Independent phases and dynamical phases are not always interchangeable.
If two energies coincide, time evolution gives their coefficients the same
phase, so their relative phase never changes. The RP ensemble nevertheless
randomizes their phases independently. It explores relative phases that the
physical trajectory never explores. This explains the former unadapted
reference's nonzero first-moment floor. The adapted reference avoids this
issue because there is only one occupied ket per block. At higher orders, equal sums of
energies can create further differences even without single-energy degeneracy.

## Sigma is the physical time-averaged density matrix

The full instantaneous density matrix contains terms

$$
\rho(t)=\sum_{\alpha,\beta}c_\alpha c_\beta^*
 e^{-i(E_\alpha-E_\beta)t}|E_\alpha\rangle\langle E_\beta|.
$$

In an infinite-time average, an oscillating term vanishes when its energy
difference is nonzero. A term with zero energy difference remains, including
off-diagonal terms between distinct eigenvectors with the same energy.

Let epsilon_b denote a **distinct energy value**. Define its projector by

$$
\Pi_b=\sum_{\alpha:E_\alpha=\epsilon_b}
|E_\alpha\rangle\langle E_\alpha|.
$$

Pi_b projects onto the whole eigenspace with that energy, not onto one
particular eigenvector or a spatial subsystem. These projectors are mutually
orthogonal, sum to the identity on the retained sector, and are unchanged by
rotating the basis within an eigenspace.

Writing rho_0=|psi_0><psi_0|, the physical infinite-time average is

$$
\sigma=\lim_{T\to\infty}\frac1T\int_0^T\rho(t)\,dt
=\sum_b\Pi_b\rho_0\Pi_b.
$$

Sigma is a fixed, generally mixed density matrix of the **full system**. It
keeps coherence inside each equal-energy block and removes coherence between
different energies. This is energy dephasing, often called the diagonal
ensemble. With degeneracies, 'diagonal' means block diagonal in an arbitrary
energy basis, not that every off-diagonal matrix element must be zero.

In an arbitrary unadapted eigenbasis, sigma and rho_RP^(1) can differ when
energies are degenerate. In the current adapted basis they agree for every
pure initial state, whether or not the spectrum is degenerate.

The former code used sum_alpha p_alpha |E_alpha><E_alpha| as the physical
dephased input. This removed within-block coherence as well. That was the
reference-state error: it did not change the Hamiltonian or the evolved ket,
but it could supply the wrong density matrix to the conditional Scrooge
construction in degenerate cases. The corrected code uses the projector formula.

## A two-level example explains the eigensolver dependence

Suppose the entire two-dimensional space has one energy, H=epsilon I, and
start in |+>=(|0>+|1>)/sqrt(2). Evolution only multiplies this ket by an overall
phase, so rho(t)=|+><+| for every t. There is only one energy projector, Pi=I,
and the physical time average is sigma=|+><+|.

If the chosen numerical energy basis is |0>,|1>, the independent RP ensemble
contains

$$
\frac{e^{i\theta_0}|0\rangle+e^{i\theta_1}|1\rangle}{\sqrt2},
$$

and its first moment is I/2. The former dephasing prescription would also
incorrectly have returned I/2. Its Frobenius distance from the actual constant
state is 1/sqrt(2).

But |+>,|-> is an equally valid energy eigenbasis. In that basis the initial
populations are 1 and 0. Independent phase randomization now leaves the same
pure-state projector unchanged, so the RP first moment is |+><+| and its
distance from the physical state is zero. The physical state and sigma have
not changed; the independently randomized reference ensemble has changed.

This is the precise meaning of the earlier phrase 'the RP reference chosen by
the numerical eigensolver': the eigensolver returns an orthonormal basis in
each degenerate space, and **the former definition** assigned an independent phase to
each vector in that returned basis. The eigensolver does not itself choose a
statistical ensemble. A more accurate phrase is 'the eigenvector-phase RP
ensemble defined using the basis returned by the eigensolver.'

This dependence concerns rotations that mix degenerate eigenvectors. Merely
changing each eigenvector's overall phase, or reordering the vectors, does not
change the RP ensemble; the coefficients transform with the basis.

The current implementation adapts this basis to the initial ket. It therefore
assigns one phase to each energy-projected component Pi_b|psi_0>, independently
of which numerical eigenbasis was originally returned. In this example it
always gives the singleton pure-state ensemble |+><+|, with zero temporal/RP
distance at every moment order. More generally, additional higher-order
energy-sum resonances can distinguish a temporal ensemble from this adapted
block-phase reference.

## How sigma becomes the projected-ensemble target

The reference density matrix sigma is an **input** to the Scrooge construction;
it is not itself the subsystem target moment shown in the projected plot.
For the computational-basis measurement outcome z on bath B, form the
unnormalized operator on A

$$
\sigma_z=\langle z|\sigma|z\rangle_B,
\qquad q_z=\mathrm{Tr}\sigma_z.
$$

Here q_z is the average outcome probability. If q_z>0, sigma_z/q_z is a
normalized density matrix on A. The generalized Scrooge target moment is

$$
G_k[\sigma]=\sum_z q_z\,S_k[\sigma_z/q_z],
$$

where S_k[R] denotes the kth moment of a single Scrooge ensemble with input R.
Zero-probability outcomes contribute zero. The projected plot compares each
state's projected moment M_k(t) with this fixed target, then averages the
distance over sampled times.

The relevant objects are therefore:

| Object | Meaning | Space |
|---|---|---|
| rho(t) | One instantaneous pure density matrix | Full system |
| sigma | Physical energy-dephased average | Full system |
| rho_RP^(k) | Moment of independent phases on occupied energy-projected components | k copies of full system |
| sigma_z | Unnormalized conditional block of sigma after a bath outcome | Subsystem A |
| G_k[sigma] | Fixed generalized Scrooge target moment | k copies of A |

The corrected implementation represents each occupied energy block using the
ket Pi_b|psi_0>/sqrt(q_b) and its weight q_b=||Pi_b|psi_0>||^2. This q_b refers
to an energy-block population; it is distinct from the bath-outcome probability
q_z above. The conditional blocks are calculated directly from these weighted
kets, avoiding storage of a full D by D sigma. Tests verify invariance under
rotations within degenerate eigenspaces.

## Understanding the former unadapted-reference plateaus

This section explains historical curves computed with independent phases in
the raw numerical eigenbasis. Its degeneracy-only floors do NOT apply to the
current adapted-reference curves. With the adapted reference this contribution
is zero at every k; higher energy-sum resonances can still give nonzero floors
for k>1. The general partial-trace inequalities above remain valid.

Let q_b=sum_(alpha in block b) p_alpha. Randomizing one phase per block gives
RP purities P_RP,k(q), whereas independent eigenvector phases give P_RP,k(p).
The nondecaying squared-error contribution from energy degeneracy alone is

$$
P_{\mathrm{RP},k}(q)-P_{\mathrm{RP},k}(p).
$$

Here purity means the trace of the square of a moment, not its trace (which is
one). To understand this subtraction, call the block-phase moment B_k and the
eigenvector-phase moment R_k. Averaging the additional relative phases inside
each block converts B_k into R_k. In a symmetric k-copy basis, this averaging
removes off-diagonal entries while keeping the diagonal entries. The remaining
and removed entries are orthogonal under the Hilbert-Schmidt inner product:

$$
\mathrm{Tr}(B_k R_k)=\mathrm{Tr}(R_k^2),\qquad
\|B_k-R_k\|_{\mathrm{HS}}^2
=\mathrm{Tr}(B_k^2)-\mathrm{Tr}(R_k^2).
$$

The normalized energy-block kets are orthogonal, with populations q_b, so
their block-phase moment has RP purity evaluated at q. The original reference
has RP purity evaluated at p. This gives the subtraction above. Extra exact
energy-sum resonances preserve still more off-diagonal entries in the temporal
moment, adding nonnegative terms to its squared distance from R_k.

This is the complete asymptotic squared distance at k=1 and also at higher
orders if no additional k-energy-sum resonances occur. Further resonances can
increase it. In the code, numerical energy grouping makes this a floor estimate
conditional on the identified blocks really representing equal energies.

For an illustrative large-D example, take D equally populated eigenvectors
in degenerate pairs. Then p has D entries 1/D and q has D/2 entries 2/D. To
leading order, a uniform RP ensemble over L components has purity k!/L^k for
fixed small k. This follows by counting the components in the symmetric
k-copy basis. If n_alpha copies occupy eigenvector alpha, with sum n_alpha=k,
the probability of that occupation pattern is

$$
w_{\boldsymbol n}=\frac{k!}{\prod_\alpha n_\alpha!}
\prod_\alpha p_\alpha^{n_\alpha},\qquad
P_{\mathrm{RP},k}(p)=\sum_{\boldsymbol n}w_{\boldsymbol n}^2.
$$

For uniform populations 1/L, the leading contribution at large L comes from
patterns with k distinct occupied vectors. There are binomial(L,k) such
patterns, each with weight k!/L^k. Their contribution is

$$
\binom Lk\left(\frac{k!}{L^k}\right)^2
=\frac{k!}{L^k}\left[1+O(L^{-1})\right].
$$

Patterns with repeated vectors give smaller contributions at fixed k as L
grows. Substituting L=D/2 for the block-phase ensemble and L=D for the
eigenvector-phase ensemble gives the squared floor

$$
\frac{k!}{(D/2)^k}-\frac{k!}{D^k}
=\frac{k!(2^k-1)}{D^k}
$$

to leading order. Taking its square root gives Frobenius floors scaling as

$$
D^{-1/2},\qquad \frac{\sqrt6}{D},\qquad
\frac{\sqrt{42}}{D^{3/2}}
$$

for k=1,2,3. This example concerns a particular population distribution; it is
not the same statement as the general partial-trace bound. A transient can
remain larger than a small floor for much longer than it remains larger than
a large floor. The lower-order curve need not flatten at the same time as
higher-order curves.

An independent NumPy construction of pure SYK4, base seed 628^3, realization
zero, even parity, gives the following estimates:

| N | Sector dimension | Detected doublets | Degeneracy-only HS floors k=1,2,3 |
|---:|---:|---:|---|
| 12 | 32 | 16 | 0.03673, 0.02012, 0.01137 |
| 20 | 512 | 256 | 0.02835, 0.004156, 0.0006450 |

The historical eigenvector-phase RP ensemble in this table uses the numerical basis
returned in that calculation. Another basis within the same degenerate blocks
can change these values without changing physical dynamics or sigma. They are
not universal SYK constants or predictions of the exact saturation times.
Higher-order resonances can increase the floors. These estimates establish
consistency of the qualitative interpretation, not a fit to the user's data.

## Numerical checks and saved diagnostics

Per-realization JSON records energy blocks, multiplicities, populations, RP
purities, grouping tolerance, estimated degeneracy floors, and the largest
observed violation of the dimension-dependent Frobenius bound. Independent
small-system tests construct the moment matrices, explicitly take partial
traces, verify trace-norm monotonicity, exhibit decreasing Frobenius floors,
and compare the overlap integral with the direct moments. The overlap/purity
identity itself passes these checks with degeneracies.

Additional tests check the adapted basis's orthonormality, reconstruction of
the initial ket, preservation of the Hamiltonian and evolved kets for exact
degeneracies, independence of the occupied kets from arbitrary block rotations,
and equality of its RP first moment with sigma. An explicit 0+2=1+1 example
checks that a genuine second-order resonance survives the basis adaptation.

The default grouping tolerance is 64*machine_epsilon times the larger of one,
spectral width, and maximum absolute energy. --degeneracy-tolerance changes it,
including zero for literal equality. Grouping nearly equal numerical energies
is an assumption about unresolved degeneracy, not a proof of an exact symmetry.
It now defines the blocks used to adapt the eigenbasis. Each block's computed
energies are replaced by their common mean before evolution and RP analysis.
This makes the rotated kets a consistent numerical eigensystem; rotating
unequal energies while keeping their old labels would not. Exact equal
energies are unchanged. Empty blocks retain their original eigenvectors.

This is a change from the earlier diagnostics-only grouping policy. It is an
explicit numerical representation of assumed exact degeneracy, and must not
be used to erase a physical splitting. Use a smaller tolerance, including zero
for literal equality, when such splittings matter. Once identified, blocks
are not grouped a second time with the original tolerance.

Let delta be the largest energy replacement. In the original eigenbasis, the
Hamiltonian change is diagonal and has operator norm delta. Consequently the
difference between the two evolved kets has norm at most |t|*delta. The JSON
field `initial_state_eigenbasis` records `max_original_block_spread`,
`max_energy_shift`, `initial_state_reconstruction_error`, and
`ket_error_bound_at_tmax`. This separates the numerical approximation from the
exact statement about true degeneracies. Finite but small resolved gaps can
still mimic a plateau over a finite observation window.

The raw cached eigensystem is retained unchanged and is adapted after loading,
so it remains independent of the initial ket. New result/intermediate names
include `rpenergyblocks` and the requested tolerance, and metadata identifies
the RP convention. The sweep plot rejects archives with differing RP conventions
or grouping tolerances. Recompute temporal curves when comparing to old results.

For pure SYK4, N=12,20,28 all have N mod 8=4, making those additional sizes useful
checks. At N=28, even parity has dimension 8192 and a single complex128 dense
matrix occupies 1 GiB before eigenvectors and solver workspace. Start with a
small simultaneous realization count when checking that case.

## Literature and the distinction between RP and Scrooge

Mark et al., [Maximum Entropy Principle in Deep Thermalization and in
Hilbert-Space Ergodicity](https://doi.org/10.1103/PhysRevX.14.041051),
Phys. Rev. X 14, 041051 (2024), explicitly discuss this reduction in Sec. IV
(following Definition 3 in the [arXiv version](https://arxiv.org/html/2403.11970v1)).
They call the remaining requirement the no-resonance condition modulo
degeneracies: apply the energy-sum condition after removing repeated energy
values, with one occupied direction per eigenspace. Their Theorem 1 relates
the temporal and RP ensembles using this condition.

First moments require no additional condition: S_1[sigma]=sigma by the
definition of a Scrooge ensemble. Equality of first moments does not establish
equality of the full ensembles. A global RP ensemble has fixed energy
populations, while a mixed-input Scrooge ensemble generally allows them to
fluctuate, so exact equality at all moments is not generic at finite dimension.

Mok, Haug, Ho and Preskill, [Nature is stingy: Universality of Scrooge ensembles
in quantum many-body systems](https://arxiv.org/abs/2601.00266) (2026),
Theorem 1 and Appendix C, prove that RP ensembles approximate Scrooge moments
under quantitative small-purity conditions. Their Appendix D addresses
normalized projected ensembles and generalized Scrooge targets. The
degeneracy reduction supplies the relevant occupied support for these
arguments; it does not remove higher-order resonances or the other assumptions
needed for Scrooge behavior.

