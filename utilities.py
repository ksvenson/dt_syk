import os
import numpy as np
import pickle
from functools import wraps
import tempfile
from typing import Literal

cache_type = Literal['npy', 'npz', 'pkl']

# Global figure settings
# LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
LEGEND_OPTIONS = {}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight', 'dpi': 800}

# directory to save all function caches
CACHE_DIR = './computation_cache/'

# directory to save all figures
FIG_DIR = './figs'

FIG_UNIT = 6
fig_size = lambda x, y: (x*FIG_UNIT, y*FIG_UNIT)


def cache(method: cache_type, base):
    """Save/reuse results under a physical-parameter note and explicit directory.

    CACHE_DIR remains the default directory. Calls may supply cache_dir=... to
    choose a different directory without changing the module's constant.
    cache_enabled=False bypasses both reads and writes. require_cache=True
    requires an existing file. These controls are consumed by this decorator;
    they are not passed to the underlying numerical function.

    The decorated function also has cache_exists(note=..., cache_dir=...) for
    checking whether a saved eigensystem is available before constructing H.
    Names identify results; inputs are not automatically compared. Different
    physics and representations therefore require different notes.

    Files are written to a temporary file in the destination directory and
    replaced atomically, so readers do not see a partially written result.
    Concurrent jobs with identical names may still duplicate computation.
    With comm supplied, all ranks agree on a hit and only rank zero writes.
    """
    def path_for(note=None, cache_dir=None):
        directory = CACHE_DIR if cache_dir is None else os.fspath(cache_dir)
        name = base if note is None else base+'_'+str(note)
        return os.path.join(directory, name+'.'+method)

    def wrap(func):
        @wraps(func)
        def inner(*args, **kwargs):
            directory = kwargs.pop('cache_dir', None)
            enabled = kwargs.pop('cache_enabled', True)
            required = kwargs.pop('require_cache', False)
            if not enabled:
                if required:
                    raise ValueError('require_cache=True conflicts with cache_enabled=False')
                return func(*args, **kwargs)
            fname = path_for(kwargs.get('note'), directory)
            comm = kwargs.get('comm')
            hit = (comm.bcast(os.path.isfile(fname) if comm.rank == 0 else None, root=0)
                   if comm is not None else os.path.isfile(fname))
            if hit:
                if method == 'npy':
                    return np.load(fname, allow_pickle=False)
                if method == 'npz':
                    with np.load(fname, allow_pickle=False) as data:
                        return dict(data)
                with open(fname, 'rb') as file:
                    return pickle.load(file)
            if required:
                raise FileNotFoundError(f'Cache required, but could not find file: {fname}')
            data = func(*args, **kwargs)
            if comm is None or comm.rank == 0:
                destination = os.path.dirname(fname) or '.'
                os.makedirs(destination, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix='.cache-', dir=destination)
                try:
                    with os.fdopen(fd, 'wb') as file:
                        if method == 'npy':
                            np.save(file, data)
                        elif method == 'npz':
                            np.savez(file, **data)
                        else:
                            pickle.dump(data, file)
                    os.replace(temporary, fname)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            if comm is not None:
                comm.Barrier()
            return data
        inner.cache_exists = lambda note=None, cache_dir=None: os.path.isfile(path_for(note, cache_dir))
        inner.cache_path = path_for
        return inner
    return wrap
