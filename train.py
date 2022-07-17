import torch
from nerf import NeRF
from dataset import load_blender
from options import GetParser
from utils import *
from nerf import Embedder
from renderer import Renderer
import random
import numpy as np
import matplotlib.pyplot as plt
from loss import MSELoss
from timeit import default_timer as timer
from tqdm import trange, tqdm
import profile

def Train(train_params, dataset, dataset_params, renderer, optimizer, LrDecay):
    it_start = train_params["it_start"]
    it_end = train_params["it_end"]
    loss_log = train_params["loss_log"]
    it_log = train_params["it_log"]
    log_dir = train_params["log_dir"]

    # print(ray_o_batch.device, ray_d_batch.device, mapped_img.device)
    bar_it = tqdm(range(it_end), leave=True, ncols=80, desc="Iteration", delay=2)
    for it in bar_it:
        if it < it_start: continue
        # bar_it.set_description("Iteration {:5d}/{:5d}".format(it+1, it_end))
        it_loss = RuntimeTrain(train_params, dataset, dataset_params, renderer, optimizer)
        LrDecay(it)
        bar_it.write("iteration:{:4d}/{:4d} loss:{:.3f}".format(it + 1, it_end + 1, it_loss))
        loss_log.append(it_loss)
        if (it+1) % it_log == 0:
            with torch.no_grad():
                RuntimeEval(train_params, dataset, dataset_params, renderer, it)
            cur_dir = os.path.join(log_dir, "it" + str(it))
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
            torch.save(renderer.fine_model.state_dict(), os.path.join(cur_dir, "fine_model.pth"))
            torch.save(renderer.coarse_model.state_dict(), os.path.join(cur_dir, "coarse_model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(cur_dir, "optimizer.pth"))
            bar_it.write("\nModel logs saved at {}".format(cur_dir))


def RuntimeTrain(train_params, dataset, dataset_params, renderer, optimizer):
    height = dataset_params["height"]
    width = dataset_params["width"]
    sample_ray = train_params["sample_ray"]
    batch_size = train_params["batch_size"]

    loss_que = []

    ray_o_batch, ray_d_batch, mapped_img = Batch2Stream(dataset, sample_ray, dataset_params)
    batch_size_rays = height * width * batch_size if sample_ray is None else sample_ray * batch_size
    bar_batch = tqdm(range(0, len(ray_o_batch), batch_size_rays), leave=False, ncols=80, desc="Training")

    for i in bar_batch:
        # bar_batch.set_description(
        #     "Processing image [{:3d}/{:3d}]".format(int(i/batch_size_rays) + 1, len(bar_batch))
        # )
        res = renderer(
            {
                "rays_o": ray_o_batch[i: i + batch_size_rays],
                "rays_d": ray_d_batch[i: i + batch_size_rays],
                "near": torch.Tensor([dataset_params["near"]]),
                "far": torch.Tensor([dataset_params["far"]])
            }
        )
        loss_fine = MSELoss(res["fine"], mapped_img[i: i + batch_size_rays])
        loss_coarse = MSELoss(res["coarse"], mapped_img[i: i + batch_size_rays])
        loss = loss_fine + loss_coarse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_numpy = loss.cpu().detach().numpy()
        loss_que.append(loss_numpy)

    return np.mean(loss_que)


def RuntimeEval(train_params, dataset, dataset_params, renderer, iteration):
    height = dataset_params["height"]
    width = dataset_params["width"]
    log_dir = train_params["log_dir"]
    choices = [0, 10, 20, 30, 40, 50]
    #bar_it = tqdm(range(len(dataset["images"])), leave=True, ncols=80, desc="Evaluating")
    bar_it = tqdm(choices, leave=True, ncols=80, desc="Evaluating")
    for it in bar_it:
        gt_image = dataset["images"][it]
        c2w = dataset["c2ws"][it]
        rays_o, rays_d = c2w2Ray(c2w, None, dataset_params)
        res = renderer(
            {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "near": torch.Tensor([dataset_params["near"]]),
                "far": torch.Tensor([dataset_params["far"]])
            }
        )
        res_image = res["fine"].cpu().numpy().reshape(height, width, 3)
        plt.subplot(2, 1, 1)
        plt.imshow(gt_image)
        plt.subplot(2, 1, 2)
        plt.imshow(res_image)
        plt.pause(2)
        plt.close()
        cur_dir = os.path.join(log_dir, "it" + str(iteration), "log_img")
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        save_dir = os.path.join(cur_dir, "{}.jpg".format(it))
        plt.imsave(save_dir, res_image)



def train_single(train_params, dataset, dataset_params, renderer, optimizer):
    height = dataset_params["height"]
    width = dataset_params["width"]
    sample_ray = train_params["sample_ray"]
    batch_size = train_params["batch_size"]
    it_start = train_params["it_start"]
    it_end = train_params["it_end"]
    loss_log = []

    # print(ray_o_batch.device, ray_d_batch.device, mapped_img.device)
    for it in range(it_start, it_end):
        sampled_idx = np.random.choice(np.arange(dataset["length"]))
        ray_o_batch, ray_d_batch, mapped_img = Batch2Stream(
            {
                "images": dataset["images"][sampled_idx].reshape((-1, height, width, 3)),
                "c2ws": dataset["c2ws"][sampled_idx].reshape((-1, 4, 4)),
            },
            sample_ray,
            dataset_params
        )
        res = renderer(
            {
                "rays_o": ray_o_batch,
                "rays_d": ray_d_batch,
                "near": torch.Tensor([dataset_params["near"]]),
                "far": torch.Tensor([dataset_params["far"]])
            },
            train=True
        )
        # print("renderer total: ", toc - tic)
        loss_fine = MSELoss(res["fine"], mapped_img)
        loss_coarse = MSELoss(res["coarse"], mapped_img)
        loss = loss_fine + loss_coarse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_numpy = loss.cpu().detach().numpy()

        train_params["LrateDecay"](it)

        # print("\riteration:{}/{} loss:{}".format(it + 1, it_end, loss_numpy))
        if it % 100 == 0:
            tqdm.write(f"[TRAIN] Iter: {it} Loss: {loss.item()}")
        loss_log.append(loss_numpy)


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

    train_set, dataset_params = load_blender("./dataset/nerf_synthetic/lego", "train", downsample=0.5)

    coarse_model = CreateModel(args)
    fine_model = CreateModel(args)

    optimizer = CreateOptimizer(args, [coarse_model, fine_model])

    loss_log = []

    if args.load_log is not None:
        if args.load_log == -1:
            dir = os.path.join(args.log_dir, args.exp_name)
            filenames = os.listdir(dir)
            filenames = [file for file in filenames if (('it' in file) and ("." not in file))]
            dir = os.path.join(dir, sorted(filenames)[-1])
        else:
            dir = os.path.join(args.log_dir, args.exp_name, "it" + str(args.log_dir))
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

    train_params = {
        "sample_ray": args.sample_ray,
        "batch_size": args.batch_size,
        "it_start": len(loss_log),
        "it_end": args.iterations,
        "loss_log": loss_log,
        "it_log": args.save_log,
        "log_dir": os.path.join(args.log_dir, args.exp_name)
    }

    Train(train_params, train_set, dataset_params, renderer, optimizer, LrDecay)


# torch.save(fine_model.state_dict(), "./fine_model.pth")
# torch.save(coarse_model.state_dict(), "./coarse_model.pth")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run()
