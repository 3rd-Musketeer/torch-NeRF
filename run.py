import random
from dataset import Blender
from nerf import Embedder
from options import GetParser
from renderer import Renderer
from utils import *
from train import Train
from torch.utils.tensorboard import SummaryWriter


def run():
    parser = GetParser()
    args = parser.parse_args()

    if args.config_dir is not None:
        args = LoadArgs(args.config_dir, args)
    if args.save_log is not None:
        SaveArgs(os.path.join(args.log_dir, args.exp_name), args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = Blender("./dataset/nerf_synthetic/lego", "train", downsample=args.down_sample)
    test_set = Blender("./dataset/nerf_synthetic/lego", "test", downsample=args.down_sample)
    val_set = Blender("./dataset/nerf_synthetic/lego", "val", downsample=args.down_sample)

    coarse_model = CreateModel(args)
    fine_model = CreateModel(args)

    optimizer = CreateOptimizer(args, [coarse_model, fine_model])

    restart_it = -1

    if args.load_log is not None:
        loaded_params = torch.load(args.load_log)
        fine_model.load_state_dict(
            torch.load(loaded_params["fine_model"])
        )
        coarse_model.load_state_dict(
            torch.load(loaded_params["coarse_model"])
        )
        optimizer.load_state_dict(
            torch.load(loaded_params["optimizer"])
        )
        restart_it = loaded_params["iteration"]

    coarse_model.to(device)
    fine_model.to(device)

    embed_pos = Embedder(args.embed_pos)
    embed_view = Embedder(args.embed_view)

    renderer = Renderer(
        args,
        {
            "device": device,
            "models": {"coarse": coarse_model, "fine": fine_model},
            "embedders": {"pos": embed_pos, "view": embed_view}
        }
    )

    LrDecay = lambda it: LrateDecay(
        it,
        args.lr,
        args.decay_rate,
        args.decay_step,
        optimizer
    )

    datasets = {"train": train_set, "test": test_set, "val": val_set}

    log_dir = os.path.join(args.log_dir, args.exp_name)

    writer = SummaryWriter(
        log_dir=log_dir,
        purge_step=restart_it,
        flush_secs=30,
    )

    train_params = {
        "sample_ray_test": args.sample_ray_test,
        "sample_ray_train": args.sample_ray_train,
        "batch_size": args.batch_size,
        "it_start": restart_it,
        "it_end": args.iterations,
        "it_log": args.save_log,
        "log_dir": log_dir,
        "idx_show": args.idx_show,
        "writer": writer,
        "shuffle": args.shuffle,
        "freq_test": args.freq_test,
        "early_downsample": args.early_downsample
    }

    Train(train_params, datasets, renderer, optimizer, LrDecay)
    writer.close()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()
