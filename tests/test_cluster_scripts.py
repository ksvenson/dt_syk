"""Cluster-script checks with fake scheduler commands; no jobs are submitted."""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ClusterScriptsTests(unittest.TestCase):
    def test_mfim_submission_and_combined_timing_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            for name in ('submit_mfim_sweep.sh','mfim_job.slurm','mfim_temp_proj.py','collect_timings.sh'):
                shutil.copy(ROOT/name,directory/name)
            # A dry run must work on a machine without the cluster venv or sbatch.
            result=subprocess.run(['bash','submit_mfim_sweep.sh','--dry-run','--L','4','6',
                                   '--','--tmax','7','--n-scr','11'],cwd=directory,
                                  text=True,capture_output=True,check=True)
            self.assertEqual(result.stdout.count('sbatch '),2)
            self.assertIn('mfim_job.slurm 4 --tmax 7 --n-scr 11',result.stdout)
            self.assertIn('mfim_job.slurm 6 --tmax 7 --n-scr 11',result.stdout)
            (directory/'.venv/bin').mkdir(parents=True)
            (directory/'.venv/bin/activate').write_text('')
            bin_dir=directory/'bin';bin_dir.mkdir()
            sbatch=bin_dir/'sbatch'
            sbatch.write_text('''#!/bin/bash
counter_file="$MFIM_TEST_JOB_COUNTER"
number=1000
[[ -f $counter_file ]] && read -r number < "$counter_file"
number=$((number+1))
printf '%s\\n' "$number" > "$counter_file"
printf 'TACC validation banner\\nSubmitted batch job %s\\n' "$number"
''')
            sbatch.chmod(0o755)
            sacct=bin_dir/'sacct'
            sacct.write_text('#!/bin/bash\nprintf "accounting %s\\n" "$*"\n')
            sacct.chmod(0o755)
            env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
                     MFIM_TEST_JOB_COUNTER=str(directory/'counter'))
            subprocess.run(['bash','submit_mfim_sweep.sh','--L','4','6','--tasks','3',
                            '--','--tmax','7'],cwd=directory,env=env,check=True,capture_output=True)
            with (directory/'mfim_manifest.csv').open() as file:rows=list(csv.DictReader(file))
            self.assertEqual([r['L'] for r in rows],['4','6'])
            self.assertEqual([r['job_id'] for r in rows],['1001','1002'])
            self.assertTrue(all(row['tasks']=='3' for row in rows))
            self.assertNotIn('g',rows[0])
            (directory/'sweep_manifest.csv').write_text(
                'N,g,tag,job_id,nodes,tasks,walltime,submitted\n8,0,N8_g0,999,1,1,1:00:00,now\n')
            (directory/'timings/N8_g0_999.csv').write_text(
                'N,g,job_id,job_name,nodes,tasks,seconds,elapsed,exit_code,finished\n8,0,999,syk,1,1,1,00:00:01,0,now\n')
            # The job wrapper must propagate analysis flags and retain a failed
            # launcher exit status in both its own exit code and the timing CSV.
            launcher=bin_dir/'ibrun'
            launcher.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$MFIM_TEST_LAUNCH_ARGS"\nexit 7\n')
            launcher.chmod(0o755)
            env.update(SLURM_SUBMIT_DIR=str(directory),SLURM_JOB_ID='1001',SLURM_JOB_NAME='mfim_L4',
                       SLURM_NNODES='1',SLURM_NTASKS='3',MFIM_TEST_LAUNCH_ARGS=str(directory/'launch_args'))
            job=subprocess.run(['bash','mfim_job.slurm','4','--tmax','7'],cwd=directory,
                               env=env,capture_output=True,text=True)
            self.assertEqual(job.returncode,7)
            self.assertEqual((directory/'launch_args').read_text().splitlines(),
                             ['python3','mfim_temp_proj.py','--L','4','--tmax','7'])
            with (directory/'timings/MFIM_L4_1001.csv').open() as file:timing=list(csv.DictReader(file))[0]
            self.assertEqual(timing['exit_code'],'7')
            collected=subprocess.run(['bash','collect_timings.sh'],cwd=directory,
                                      env=env,text=True,capture_output=True,check=True)
            self.assertIn('mfim_manifest.csv',collected.stdout)
            self.assertIn('sweep_manifest.csv',collected.stdout)
            self.assertIn('1001,1002',collected.stdout)
            self.assertIn('MFIM_L6',collected.stdout) # Submitted but no timing file.
            self.assertIn('L,model',collected.stdout)
            self.assertIn('N,g',collected.stdout)


if __name__=='__main__':unittest.main()
