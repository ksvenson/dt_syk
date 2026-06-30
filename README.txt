Instructions for installing dynamite locally
(Not guaranteed to work, some machines may have their own quirks)

2. Follow instructions here: https://dynamite.readthedocs.io/en/latest/install.html, up
   until, but *not* including, the "Building Dynamite" section.

   Note: while building `petsc`, MUMPS may throw an exception. The fix is to manually
   add
   ```
   output1 = err1 = output2 = err2 = output3 = err3 = ''
   ```
   to
   `<project_dir>/petsc/config/BuildSystem/config/packages/MUMPS.py`
   after line 171. `<project_dir>/petsc/configure.log` gives some extra info.

3. double-check that environment variables are set:
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

4. Run `uv sync`
