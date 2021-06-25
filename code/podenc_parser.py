import argparse
import os


def parse_arguments():
    """Read commandline arguments

    Returns:
        Namespace: input as well as default arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--project-id', type=str, default='podcast')
    parser.add_argument('--jar-name', type=str, default=None)
    parser.add_argument('--conversation-id', type=int, default=0)

    parser.add_argument('--word-value',
                        default='all',
                        choices=['all', 'top', 'bottom'])
    parser.add_argument('--window-size', type=int, default=200)
    parser.add_argument('--stim', type=str, default='Podcast')
    parser.add_argument('--pilot', type=str, default='')
    parser.add_argument('--lags', nargs='+', type=int)
    parser.add_argument('--output-prefix', type=str, default='')
    parser.add_argument('--nonWords', action='store_true', default=False)
    parser.add_argument('--datum-emb-fn',
                        type=str,
                        default='podcast-datum-glove-50d.csv')
    parser.add_argument('--gpt2', type=int, default=1)
    parser.add_argument('--bert', type=int, default=None)
    parser.add_argument('--bart', type=int, default=None)
    parser.add_argument('--glove', type=int, default=1)
    parser.add_argument('--electrodes', nargs='*', type=int)
    parser.add_argument('--npermutations', type=int, default=1)
    parser.add_argument('--min-word-freq', nargs='?', type=int, default=1)
    parser.add_argument('--job-id', type=int, default=0)
    parser.add_argument('--output-parent-dir', type=str, default='test')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sid', nargs='?', type=int, default=None)
    group.add_argument('--sig-elec-file', nargs='?', type=str, default=None)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--shuffle', action='store_true', default=False)
    group1.add_argument('--phase-shuffle', action='store_true', default=False)

    parser.add_argument('--replication', action='store_true', default=False)
    parser.add_argument('--fold-idx', type=int, default=None)

    args = parser.parse_args()

    if not args.sid and args.electrodes:
        parser.error("--electrodes requires --sid")

    if not args.shuffle and not args.phase_shuffle:
        args.npermutations = 1

    if args.sig_elec_file:
        args.sid = 777

    if os.getenv("SLURM_ARRAY_TASK_ID", None):
        args.output_parent_dir += str(os.getenv("SLURM_ARRAY_TASK_ID"))

    return args
