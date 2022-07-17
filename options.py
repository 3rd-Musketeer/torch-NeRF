import argparse


def GetParser():
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--exp_name', type=str, required=True)

    parser.add_argument('--log_dir', type=str, default='./log/')

    parser.add_argument("--random_seed", type=int,
                        default=829)
    # dataset parameters
    parser.add_argument('--dataset', type=str,
                        default="blender")
    parser.add_argument('--shuffle',
                        action='store_true')

    # renderer parameters
    parser.add_argument('--sample_coarse', type=int,
                        default=64)
    parser.add_argument('--sample_fine', type=int,
                        default=128)
    parser.add_argument('--sample_ray', type=int,
                        default=1024)
    parser.add_argument('--ray_chunk', type=int,
                        default=1024*16)
    parser.add_argument('--ray_batch', type=int,
                        default=1024*32)
    parser.add_argument('--rand_sample', type=bool,
                        default=True)

    # mlp parameters
    parser.add_argument('--inPos_ch', type=int,
                        default=3)
    parser.add_argument('--inView_ch', type=int,
                        default=3)
    parser.add_argument('--pos_branch', type=list,
                        default=[4])
    parser.add_argument('--out_ch', type=int,
                        default=4)
    parser.add_argument('--net_width', type=int,
                        default=256)
    parser.add_argument('--hidden_depth', type=int,
                        default=8)
    parser.add_argument('--embed_pos', type=int,
                        default=10)
    parser.add_argument('--embed_view', type=int,
                        default=4)

    # training parameters
    parser.add_argument('--batch_size', type=int,
                        default=1)
    parser.add_argument('--iterations', type=int,
                        default=500)
    parser.add_argument('--lr', type=float,
                        default=5e-4)
    parser.add_argument('--decay_step', type=int,
                        default=250)
    parser.add_argument("--decay_rate", type=float,
                        default=0.1)
    parser.add_argument("--config_dir", type=str,
                        default=None)
    parser.add_argument("--save_log", type=int,
                        default=10)

    parser.add_argument('--load_log', type=int,
                        default=None)
    return parser


if __name__ == '__main__':
    parser = GetParser()
    print(parser.parse_args())
