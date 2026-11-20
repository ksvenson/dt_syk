"""Compare the four MFIM distance curves across independently scheduled sizes."""
import argparse
from plot_N_g_sweep import plot_archives


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',nargs='+',required=True,help='MFIM results_*.npz archives')
    parser.add_argument('--gap-lines',action='store_true')
    parser.add_argument('--thermalization-fraction',type=float)
    args=parser.parse_args()
    if args.thermalization_fraction is not None and not 0<args.thermalization_fraction<1:
        parser.error('--thermalization-fraction must lie between zero and one')
    plot_archives(args.results,args.gap_lines,args.thermalization_fraction,expected_model='MFIM')
