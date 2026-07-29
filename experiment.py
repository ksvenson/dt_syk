import pickle as pkl


class Experiment():
    def __init__(
        self,
        L,
        H_str,
        psi0_str,
        k_series,
        temp_log_tmax=None,
        temp_log_ns=None,
        hx=None,
        hz=None,
    ):
        self.L = L
        self.H_str = H_str
        self.psi0_str = psi0_str
        self.k_series = k_series
        self.temp_log_tmax = temp_log_tmax
        self.temp_log_ns = temp_log_ns
        self.hx = hx
        self.hz = hz
        self.cache_notes = {}
        
        if H_str.lower() == 'mfim':
            assert hx is not None
            assert hz is not None
            self.cache_notes['H'] = f'L{L}_{H_str}_hx{hx:.4f}_hz{hz:.4f}'
        elif H_str.lower() == 'syk':
            # TODO:
            raise NotImplementedError()
        else:
            raise NotImplementedError('Unsupported Hamiltonian.')

        self.psi0_note = None
        if psi0_str.lower() == 'mark37':
            self.cache_notes['psi0'] = f'L{L}_psi0{psi0_str}'
        else:
            raise NotImplementedError('Unsupported initial state.')

    def exact_temp_rp_2norm_note(self, k):
        assert self.temp_log_tmax is not None
        assert self.temp_log_ns is not None
        return self.cache_notes['H'] + f'_{self.psi0_note}_k{k}_log-tmax{self.temp_log_tmax}_log-ns{self.temp_log_ns}'
    
    def save(self, fname='./current_experiment.pkl'):
        with open(fname, 'wb') as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, fname='./current_experiment.pkl'):
        with open(fname, 'rb') as f:
            exp = pkl.load(f)
        return exp

