import random
from dataset import blender
from nerf import Embedder
from options import GetParser
from renderer import Renderer
from utils import *
from train import Train
from torch.utils.tensorboard import SummaryWriter


def run():
    parser = GetParser()
    args = parser.parse_args()

    if args.load_log is not None:
        args = LoadArgs(os.path.join(args.log_dir, args.exp_name), args)
    if args.save_log is not None:
        SaveArgs(os.path.join(args.log_dir, args.exp_name), args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = blender("./dataset/nerf_synthetic/lego", "train", downsample=args.down_sample)
    test_set = blender("./dataset/nerf_synthetic/lego", "test", downsample=args.down_sample)
    val_set = blender("./dataset/nerf_synthetic/lego", "val", downsample=args.down_sample)

    coarse_model = CreateModel(args)
    fine_model = CreateModel(args)

    optimizer = CreateOptimizer(args, [coarse_model, fine_model])

    loss_log = []

    restart_it = None

    if args.load_log is not None:
        if args.load_log == -1:
            dir = os.path.join(args.log_dir, args.exp_name)
            filenames = os.listdir(dir)
            filenames = [file for file in filenames if (('it' in file) and ("." not in file))]
            dir = os.path.join(dir, sorted(filenames)[-1])
            restart_it = int(sorted(filenames)[-1][2:])
        else:
            dir = os.path.join(args.log_dir, args.exp_name, str(args.log_dir) + "it")
        assert os.path.exists(dir)
        fine_model.load_state_dict(
            os.path.join(dir, "fine_model.pth")
        )
        coarse_model.load_state_dict(
            os.path.join(dir, "coarse_model.pth")
        )
        optimizer.load_state_dict(
            os.path.join(dir, "optimizer.pth")
        )
        loss_log = torch.load(
            os.path.join(dir, "loss_log.pth")
        )

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
        "it_start": len(loss_log),
        "it_end": args.iterations,
        "loss_log": loss_log,
        "it_log": args.save_log,
        "log_dir": log_dir,
        "idx_show": args.idx_show,
        "writer": writer,
        "shuffle": args.shuffle
    }

    Train(train_params, datasets, renderer, optimizer, LrDecay)
    writer.close()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()
