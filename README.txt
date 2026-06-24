Instructions for installing dynamite:
1. Follow instructions here: https://dynamite.readthedocs.io/en/latest/install.html, up
   until, but *not* including, the "Building Dynamite" section.

   Note: while building `petsc`, MUMPS may throw an exception. The fix is to manually
   add
   ```
   output1 = err1 = output2 = err2 = output3 = err3 = ''
   ```
   to
   `<project_dir>/petsc/config/BuildSystem/config/packages/MUMPS.py`
   after line 171. `<project_dir>/petsc/configure.log` gives some extra info.

2. double-check that environment variables are set:
   ```
   export PETSC_DIR="<project_dir>/petsc"
   export PETSC_ARCH="complex-opt"
   export SLEPC_DIR="<project_dir>/slepc"
   ```
   Or use `direnv` and put this into `<project_dir>/.envrc`:
   ```
   export PROJECT_ROOT="$(expand_path .)"
   export PETSC_DIR="$PROJECT_ROOT/petsc"
   export PETSC_ARCH="complex-opt"
   export SLEPC_DIR="$PROJECT_ROOT/slepc"
   ```

3. Run `uv sync`
