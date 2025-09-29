import argparse


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    # Humanoid-v4， HalfCheetah-v4
    parser.add_argument('--env-name', default="HalfCheetah-v4", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                        help='damping (default: 1e-1)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--start-safety', type=int, default=40, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--safety-bound', type=float, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument("--with-momentum", default="False", help='Whether agent uses momentum to update the policy')
    # args = parser.parse_args()
    parser.add_argument("--exp_name", '-e', type=str, default=None)
    parser.add_argument("--steps", type=int, default=2500000)
    parser.add_argument("--train_rl", action='store_true', default=False)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--obs_w", type=float, default=3.0)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--rover_vmax", type=float, default=10.0)
    parser.add_argument("--close_thres", type=float, default=0.8)
    parser.add_argument("--hold_t", type=int, default=3)
    parser.add_argument("--nt", type=int, default=10)
    parser.add_argument("--hard_soft_step", action='store_true', default=False)
    parser.add_argument("--norm_ap", action='store_true', default=False)
    parser.add_argument("--tanh_ratio", type=float, default=0.05)
    parser.add_argument("--rover_vmin", type=float, default=0.0)
    parser.add_argument("--battery_charge", type=float, default=5.0)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=250000)
    return parser