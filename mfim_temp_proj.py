"""Periodic mixed-field Ising analysis with MPI ranks assigned time intervals."""
import argparse
import json
import os
from time import perf_counter

import numpy as np
import dynamite as dm
import dynamite.tools as tools

import ensemble_analysis as ea
import hamiltonian_factory as hf
import linalg as la
import projected_funcs as pf
import temporal_funcs as tf
import thermalization as th

# Kim, Ikeda and Huse, Phys. Rev. E 90, 052105 (2014), Eq. (7).
HX, HZ, J = .9045, .8090, 1.


def parse_args():
    """Read spin count L, subsystem spin count LA, and shared analysis controls.

    L and LA count spin-1/2 sites, unlike the Majorana counts N and NA in SYK.
    Couplings and periodic boundaries are fixed, with no coupling sweep and no
    disorder realizations. Seed controls only time and Scrooge sampling. All MPI
    ranks analyze the same initial computational vacuum and Hamiltonian.
    """
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--L',type=ea.positive_int,required=True,help='Number of spin-1/2 sites, at least 3')
    parser.add_argument('--LA',type=ea.positive_int,default=2,help='Subsystem spins; DA=2**LA')
    ea.add_sampling_arguments(parser)
    args = parser.parse_args()
    if args.L < 3 or args.LA > args.L:
        parser.error('Need L>=3 and 1<=LA<=L')
    ea.validate_sampling_arguments(parser,args)
    return args


def time_slice(length, size, rank):
    """Assign contiguous time indices, including empty assignments when needed."""
    count,remainder = divmod(length,size)
    begin = rank*count+min(rank,remainder)
    return slice(begin,begin+count+(rank<remainder))


def broadcast_array(comm, values):
    """Share a NumPy array without pickling the large eigenvector matrix.

    Rank zero supplies values. Other ranks allocate the same shape and dtype,
    then MPI broadcasts the numerical buffer. Each rank has one local copy;
    this is not a distributed eigensolver or a shared-memory matrix.
    """
    header = (values.shape,values.dtype.str) if comm.Get_rank()==0 else None
    shape,dtype = comm.bcast(header,root=0)
    values = np.ascontiguousarray(values) if comm.Get_rank()==0 else np.empty(shape,dtype=dtype)
    comm.Bcast(values,root=0)
    return values


def gather_rows(comm, values):
    """Collect contiguous chronological rows from time chunks on rank zero."""
    rows = comm.gather(len(values),root=0)
    if comm.Get_rank()==0:
        result = np.empty((sum(rows),)+values.shape[1:],dtype=values.dtype)
        stride = int(np.prod(values.shape[1:]))
        counts = np.array(rows,dtype=np.int64)*stride
        offsets = np.r_[0,np.cumsum(counts[:-1])]
        receive = [result,(counts,offsets)]
    else:
        result,receive = None,None
    comm.Gatherv(np.ascontiguousarray(values),receive,root=0)
    return result


def collective_errors(comm, error):
    """Raise on every rank before collection if any rank's local work failed."""
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        raise RuntimeError('; '.join(f'rank {i}: {msg}' for i,msg in enumerate(errors) if msg is not None))


def main():
    """Compute the same four curves as SYK for one deterministic periodic chain.

    Rank zero diagonalizes once and builds the fixed generalized Scrooge target.
    All ranks then integrate different temporal intervals and evolve different
    sampled states. Scalar interval integrals are collected before constructing
    temporal moment errors. Per-state projected distances are collected before
    their cumulative averages. There is one physical trajectory, not an average
    over disorder or duplicate Hamiltonians. Archive realization axes have size 1.
    """
    args = parse_args()
    comm = tools.MPI_COMM_WORLD().tompi4py()
    rank,size = comm.Get_rank(),comm.Get_size()
    started = perf_counter()
    os.makedirs(args.cache_dir,exist_ok=True)
    dm.config.L = args.L
    D,DA = 2**args.L,2**args.LA
    ks = np.arange(1,args.k_max+1)
    times = np.linspace(0,args.tmax,max(1,int(np.ceil(args.tmax/args.projected_dt)))+1)
    sample_times = ea.projected_times(times,args.projected_sampling,args.seed,args.time_seed)
    state_part = time_slice(len(times),size,rank)
    interval_part = time_slice(len(times)-1,size,rank)
    h_note = f'MFIM_L{args.L}_hx{HX}_hz{HZ}_J{J}_periodic_full'
    run_note = (f'MFIM_L{args.L}_LA{args.LA}_seed{args.seed}_T{args.tmax}_dt{args.dt}'
                f'_pdt{args.projected_dt}_ps{args.projected_sampling}_ts{args.time_seed}'
                f'_scr{args.n_scr}_k{args.k_max}_tol{args.degeneracy_tolerance}'
                f'_check{int(args.check_dt)}_gap{args.gap_k_max}_gcap{args.gap_max_sums}'
                f'_fraction{args.thermalization_fraction}')

    # 1. Diagonalize ONCE on rank zero; reuse the raw eigensystem cache when warm.
    # No fermion-parity restriction: the transverse field flips Z-basis spins.
    error = None
    if rank==0:
        try:
            print(f'MFIM L={args.L}, DA={DA}; {size} MPI ranks split time intervals; '
                  f'J={J}, hx={HX}, hz={HZ}, periodic boundaries',flush=True)
            stage = perf_counter()
            eig_hit = args.cache_eigensystem and la.eig_system.cache_exists(note=h_note,cache_dir=args.cache_dir)
            if eig_hit:
                h_seconds = 0.
                eigs = la.eig_system(None,note=h_note,cache_dir=args.cache_dir,require_cache=True)
            else:
                H = hf.MFIM_numpy(HX,HZ,J,boundary='periodic',note=h_note,
                    cache_dir=args.cache_dir,cache_enabled=args.cache_hamiltonian)
                h_seconds = perf_counter()-stage
                eigs = la.eig_system(H,note=h_note,cache_dir=args.cache_dir,
                                    cache_enabled=args.cache_eigensystem)
                del H
            eig_seconds = perf_counter()-stage-h_seconds

            # 2. Rotate the eigenbasis once, keeping the coherent vacuum intact.
            psi0 = np.zeros(D,complex); psi0[0] = 1
            prepared = ea.prepare_eigensystem(eigs,psi0,ks,args)
            evals,pops = prepared['evals'],prepared['pops']
            diagnostics = prepared['diagnostics']
            dt = prepared['dt']
            active = pops>0
            vectors = np.ascontiguousarray(np.real_if_close(prepared['evecs'][:,active]))
            coefficients = prepared['coefficients'][active]
            occupied = evals[active]
            occupied_pops = pops[active]
            del eigs,prepared

            # 3. Build the fixed generalized Scrooge target from the same basis.
            stage = perf_counter()
            target = pf.cond_scr_moment(vectors,occupied_pops,args.n_scr,ks,DA,
                np.random.default_rng(np.random.SeedSequence(args.seed,spawn_key=(0,1))),
                note=f'{h_note}_vacuum_DA{DA}_seed{args.seed}_scr{args.n_scr}_k{args.k_max}_tol{args.degeneracy_tolerance}',
                symmetric=True,cache_dir=args.cache_dir)
            scrooge_seconds = perf_counter()-stage
            gaps,references = np.full(len(ks),np.nan),np.full(len(ks),np.nan)
            diagnostics['moment_gaps'] = []
            for j,k in enumerate(ks):
                if k <= args.gap_k_max:
                    gap = th.minimum_moment_gap(occupied,int(k),args.gap_max_sums)
                    gap['k'] = int(k); diagnostics['moment_gaps'].append(gap)
                    if gap['min_gap'] is not None:gaps[j] = gap['min_gap']
                    if gap['reference_time'] is not None:references[j] = gap['reference_time']
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = comm.bcast(error,root=0)
    if error is not None:
        raise RuntimeError('MFIM preparation failed: '+error)

    # Share the occupied eigensystem and small target moments, not a second H.
    dt = comm.bcast(dt if rank==0 else None,root=0)
    occupied = broadcast_array(comm,occupied if rank==0 else None)
    occupied_pops = broadcast_array(comm,occupied_pops if rank==0 else None)
    vectors = broadcast_array(comm,vectors if rank==0 else None)
    coefficients = broadcast_array(comm,coefficients if rank==0 else None)
    target = {f'k{k}':broadcast_array(comm,target[f'k{k}'] if rank==0 else None) for k in ks}

    # 4. Integrate local time intervals and compute per-state distances locally.
    # All ranks use identical quadrature spacing and global sample-index seeds.
    local_error = None
    try:
        stage = perf_counter()
        edges = times[interval_part.start:interval_part.stop+1]
        integrals = tf.temporal_overlap_integrals(occupied,occupied_pops,ks,edges,dt,
                                                  reporting_spacing=times[1])
        refined = (tf.temporal_overlap_integrals(occupied,occupied_pops,ks,edges,dt/2,
                    reporting_spacing=times[1]) if args.check_dt else None)
        temporal_seconds = perf_counter()-stage
        stage = perf_counter()
        sampled = pf.time_evolved_overlaps_and_ps_2norm(occupied,vectors,coefficients,
            sample_times[state_part],ks,DA,target,chunk=args.chunk,symmetric=True,
            instantaneous=True,n_scr=args.n_scr,rng=np.random.default_rng(args.seed),
            sample_seed=args.seed,sample_indices=np.arange(state_part.start,state_part.stop),
            note=f'{run_note}_times{state_part.start}to{state_part.stop}',
            cache_dir=args.cache_dir,cache_enabled=args.cache_projected)
        projected_seconds = perf_counter()-stage
        if args.verbose:
            print(f'Rank {rank}: temporal intervals {interval_part.start}:{interval_part.stop}; '
                  f'projected states {state_part.start}:{state_part.stop}; '
                  f'{temporal_seconds:.2f}s temporal, {projected_seconds:.2f}s projected',flush=True)
    except Exception as exc:
        local_error = f'{type(exc).__name__}: {exc}'
    collective_errors(comm,local_error)

    # 5. Collect chronological intervals/states. Average only AFTER collection.
    integral_f = gather_rows(comm,integrals['integral_f'])
    integral_uf = gather_rows(comm,integrals['integral_uf'])
    if args.check_dt:
        refined_f = gather_rows(comm,refined['integral_f'])
        refined_uf = gather_rows(comm,refined['integral_uf'])
    projected = gather_rows(comm,sampled['2norm'])
    single_raw = gather_rows(comm,sampled['single_scr_2norm'])
    rank_timings = comm.gather({'rank':rank,'intervals':interval_part.stop-interval_part.start,
        'states':state_part.stop-state_part.start,'temporal_seconds':temporal_seconds,
        'projected_seconds':projected_seconds},root=0)
    if rank!=0:
        return None

    # 6. Construct the global temporal moments and the two running averages.
    h = times[1]/max(2,2*int(np.ceil(times[1]/(2*dt))))
    temporal = tf.temporal_distance_from_integrals(occupied_pops,ks,times,
                       integral_f,integral_uf,h)
    ea.temporal_checks(evals,pops,ks,temporal['distance'],diagnostics)
    diagnostics['dt_effective'] = float(h)
    if args.check_dt:
        h_half = times[1]/max(2,2*int(np.ceil(times[1]/dt)))
        fine = tf.temporal_distance_from_integrals(occupied_pops,ks,times,
                                                   refined_f,refined_uf,h_half)
        diagnostics['dt_half_max_squared_error'] = np.max(abs(
            temporal['distance_squared']-fine['distance_squared']),axis=0).tolist()
        diagnostics['dt_half_max_distance_error'] = np.max(abs(
            temporal['distance']-fine['distance']),axis=0).tolist()
    temporal_values = temporal['distance'][None]
    projected_average = ea.running_distance(projected,args.projected_sampling)[None]
    single_average = ea.running_distance(single_raw,args.projected_sampling)[None]
    single_instant = single_raw[None]
    initial_state = {'basis':'full computational basis','basis_labels':[0],
                     'amplitudes_real':[1.],'amplitudes_imag':[0.]}
    metadata = dict(vars(args),model='MFIM',hx=HX,hz=HZ,J=J,boundary='periodic',
        parity='full',realizations=1,hamiltonian_convention='J*sum ZZ+hx*sum X+hz*sum Z',
        rp_reference='initial-state-adapted energy blocks',initial_state=initial_state,
        single_scrooge_evaluation='numerical moment difference',
        energy_grouping='mean energy within each numerically degenerate block',
        mpi_partition='contiguous time intervals; global sample-index Scrooge streams')
    wall = perf_counter()-started
    result_path = os.path.join(args.cache_dir,'results_'+run_note+'.npz')
    np.savez_compressed(result_path,tau_series=times,k_series=ks,
        temp_rp_2norm=temporal_values,proj_scr_2norm=projected_average,
        proj_single_scr_2norm=single_average,proj_single_scr_instant_2norm=single_instant,
        projected_sample_times=sample_times[None],moment_min_gaps=gaps[None],
        moment_gap_times=references[None],sampling_bound_warnings=np.array([bool(diagnostics['warnings'])]),
        analysis_wall_seconds=np.array(wall),metadata=np.array(json.dumps(metadata)),
        temporal_threshold_times=ea.observed_times(times,temporal_values,ks,args.thermalization_fraction),
        projected_threshold_times=ea.observed_times(times,projected_average,ks,args.thermalization_fraction),
        single_scrooge_threshold_times=ea.observed_times(times,single_average,ks,args.thermalization_fraction))
    timing = {'hamiltonian_seconds':h_seconds,'diagonalization_seconds':eig_seconds,
        'temporal_seconds':max(row['temporal_seconds'] for row in rank_timings),
        'projected_seconds':max(row['projected_seconds'] for row in rank_timings),
        'scrooge_seconds':scrooge_seconds,'total_seconds':wall,'eigensystem_cache_hit':eig_hit,
        'rank_timings':rank_timings,'parameters':metadata,'initial_state':initial_state,
        'sampling_diagnostics':diagnostics}
    timing_path = os.path.join(args.cache_dir,'timing_'+run_note+'.json')
    with open(timing_path,'w') as file:json.dump(timing,file,indent=2)
    print(f'Analysis complete: MFIM L={args.L} in {wall:.1f}s',flush=True)
    print(f'Results saved: {result_path}',flush=True)
    print(f'Detailed timings and diagnostics: {timing_path}',flush=True)
    if diagnostics['warnings']:
        print('WARNING: temporal grid exceeds the sampling bound; see diagnostics.',flush=True)
    if not args.no_plots:
        ea.plot_results(times,temporal_values,projected_average,ks,run_note,
            f'MFIM L={args.L}, periodic, hx={HX}, hz={HZ}, J={J}',
            gap_times=references[None] if args.gap_lines else None,
            fraction=args.thermalization_fraction,single=single_average,
            single_instant=single_instant,sample_times=sample_times[None])
    return result_path


if __name__=='__main__':
    main()
