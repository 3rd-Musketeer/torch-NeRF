import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from loss import MSELoss, PSNR
from utils import *


def Train(train_params, datasets, renderer, optimizer, LrDecay):
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    val_dataset = datasets["val"]
    it_start = train_params["it_start"]
    it_end = train_params["it_end"]
    loss_log = train_params["loss_log"]
    it_log = train_params["it_log"]
    log_dir = train_params["log_dir"]
    writer = train_params["writer"]

    # print(ray_o_batch.device, ray_d_batch.device, mapped_img.device)
    bar_it = tqdm(range(it_end), leave=True, ncols=80, desc="Iteration", delay=2)
    for it in bar_it:
        if it < it_start: continue
        # bar_it.set_description("Iteration {:5d}/{:5d}".format(it+1, it_end))
        train_loss, train_psnr = RuntimeTrain(train_params, train_dataset, renderer, optimizer)
        writer.add_scalar('Loss/train', train_loss, it + 1)
        writer.add_scalar('PSNR/train', train_psnr, it + 1)
        LrDecay(it)
        bar_it.write("\nIteration:{:4d}/{:4d} train_loss:{:.3f} train_psnr:{:.3f}".format(
            it + 1, it_end, train_loss, train_psnr)
        )
        loss_log.append(train_loss)
        with torch.no_grad():
            test_loss, test_psnr = RuntimeTest(train_params, test_dataset, renderer, it)
        writer.add_scalar('Loss/test', test_loss, it + 1)
        writer.add_scalar('PSNR/test', test_psnr, it + 1)
        bar_it.write("-" * 10 + "Test Results" + "-" * 10)
        bar_it.write("test_loss:{:.3f} test_psnr:{:.3f}".format(test_loss, test_psnr))
        bar_it.write("-" * 32)
        if (it + 1) % it_log == 0 or it == len(bar_it) - 1:
            with torch.no_grad():
                val_loss, val_psnr = RuntimeEval(train_params, val_dataset, renderer, it)
            bar_it.write("-" * 10 + "Eval Results" + "-" * 10)
            bar_it.write("eval_loss:{:.3f} eval_psnr:{:.3f}".format(val_loss, val_psnr))
            bar_it.write("-" * 32)
            writer.add_scalar('Loss/val', val_loss, it + 1)
            writer.add_scalar('PSNR/val', val_psnr, it + 1)
            cur_dir = os.path.join(log_dir, "it" + str(it + 1))
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
            torch.save(renderer.fine_model.state_dict(), os.path.join(cur_dir, "fine_model.pth"))
            torch.save(renderer.coarse_model.state_dict(), os.path.join(cur_dir, "coarse_model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(cur_dir, "optimizer.pth"))
            bar_it.write("Model logs saved at {}".format(cur_dir))
        writer.flush()


def RuntimeTrain(train_params, dataset, renderer, optimizer):
    dataset_params = dataset.params
    height = dataset_params["height"]
    width = dataset_params["width"]
    sample_ray = train_params["sample_ray_train"]
    batch_size = train_params["batch_size"]
    shuffle = train_params["shuffle"]

    if shuffle:
        dataset.shuffle()

    loss_que = []
    psnr_que = []

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
        psnr_fine = PSNR(res["fine"], mapped_img[i: i + batch_size_rays])
        loss = loss_fine + loss_coarse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_numpy = loss.cpu().detach().numpy()
        loss_que.append(loss_numpy)
        psnr_que.append(psnr_fine.cpu().detach().numpy())

    return np.mean(loss_que), np.mean(psnr_que)


def RuntimeTest(train_params, dataset, renderer, iteration):
    dataset_params = dataset.params
    height = dataset_params["height"]
    width = dataset_params["width"]
    log_dir = train_params["log_dir"]
    sample_ray = train_params["sample_ray_test"]
    choices = train_params["idx_show"]

    choices = choices if choices is not None else list(range(len(dataset)))

    loss_que = []
    psnr_que = []
    # bar_it = tqdm(range(len(dataset["images"])), leave=True, ncols=80, desc="Evaluating")
    bar_it = tqdm(range(len(dataset)), leave=False, ncols=80, desc="Evaluating")
    for it in bar_it:
        gt_image = dataset[it]["images"].reshape(-1, 3)
        c2w = dataset[it]["c2ws"]
        rays_o, rays_d, pos = c2w2Ray(c2w, sample_ray, dataset_params)
        gt_image = gt_image[pos.cpu().detach().numpy()]
        res = renderer(
            {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "near": torch.Tensor([dataset_params["near"]]),
                "far": torch.Tensor([dataset_params["far"]])
            }
        )
        res_image = res["fine"].cpu().numpy().reshape(-1, 3)
        loss_numpy = np.mean((res_image - gt_image) ** 2)
        psnr = 10 * np.log10(1.0 / loss_numpy)
        loss_que.append(loss_numpy)
        psnr_que.append(psnr)

        # if it in choices:
        #     plt.subplot(2, 1, 1)
        #     plt.imshow(gt_image)
        #     plt.subplot(2, 1, 2)
        #     plt.imshow(res_image)
        #     plt.pause(2)
        #     plt.close()
        # cur_dir = os.path.join(log_dir, "it" + str(iteration), "log_img")
        # if not os.path.exists(cur_dir):
        #     os.makedirs(cur_dir)
        # save_dir = os.path.join(cur_dir, "{}.png".format(it + 1))
        # plt.imsave(save_dir, res_image)

    return np.mean(loss_que), np.mean(psnr_que)


def RuntimeEval(eval_params, dataset, renderer, iteration):
    dataset_params = dataset.params
    height = dataset_params["height"]
    width = dataset_params["width"]
    log_dir = eval_params["log_dir"]
    choices = eval_params["idx_show"]
    writer = eval_params["writer"]

    choices = choices if choices is not None else list(range(len(dataset)))

    loss_que = []
    psnr_que = []
    # bar_it = tqdm(range(len(dataset["images"])), leave=True, ncols=80, desc="Evaluating")
    bar_it = tqdm(range(len(dataset)), leave=True, ncols=80, desc="Evaluating")
    for it in bar_it:
        gt_image = dataset[it]["images"]
        c2w = dataset[it]["c2ws"]
        rays_o, rays_d, _ = c2w2Ray(c2w, None, dataset_params)
        res = renderer(
            {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "near": torch.Tensor([dataset_params["near"]]),
                "far": torch.Tensor([dataset_params["far"]])
            }
        )
        res_image = res["fine"].cpu().numpy().reshape(height, width, 3)
        loss_numpy = np.mean((res_image - gt_image) ** 2)
        psnr = 10 * np.log10(1.0 / loss_numpy)
        loss_que.append(loss_numpy)
        psnr_que.append(psnr)
        if it in choices:
            plt.subplot(2, 1, 1)
            plt.imshow(gt_image)
            plt.subplot(2, 1, 2)
            plt.imshow(res_image)
            plt.pause(2)
            plt.close()
        cur_dir = os.path.join(log_dir, "it" + str(iteration), "log_img")
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        save_dir = os.path.join(cur_dir, "{}.png".format(it + 1))
        plt.imsave(save_dir, res_image)
        writer.add_image('Image/Prediction', res_image, iteration + 1, dataformats="HWC")
        writer.add_image('Image/Ground Truth', gt_image, iteration + 1, dataformats="HWC")

    return np.mean(loss_que), np.mean(psnr_que)


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
