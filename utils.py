import torch
import numpy as np
import nerf
import os
import json


def SaveArgs(dir, args):
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'args.txt')
    # if os.path.exists(dir):
    #     raise "[SaveArgs] log exists!"
    f = open(os.path.join(dir), 'w')
    json.dump(args.__dict__, f, indent=2)
    f.close()


def LoadArgs(dir, args):
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'args.txt')
    if not os.path.exists(dir):
        raise "[LoadArgs] log not found!"
    with open(dir, 'r') as f:
        args.__dict__ = json.load(f)
    return args


def LrateDecay(it, lrate, drate, dstep, optimizer):
    new_lrate = lrate * (drate ** (it / dstep))
    for params in optimizer.param_groups:
        params["lr"] = new_lrate
    return new_lrate


def Vec2Img(input_vec):
    return 255.0 * np.clip(input_vec, 0, 1).astype(np.uint8)


def Batch2Stream(img_batch, sample_ray, params):
    N = len(img_batch)
    height = params["height"]
    width = params["width"]
    focal = params["focal"]
    ray_o_batch = []  # [B, N_rays, 3]
    ray_d_batch = []  # [B, N_rays, 3]
    mapped_img = []  # [B, N_rays, 1]
    for i in range(N):
        img = torch.Tensor(img_batch[i]["images"])
        c2w = torch.Tensor(img_batch[i]["c2ws"])
        rays_o, rays_d, pos = GetRays(height, width, focal, c2w, sample_ray)  # [N_rays, 3] [N_rays, 1]
        ray_o_batch.append(rays_o)
        ray_d_batch.append(rays_d)
        mapped_img.append(
            img.view(-1, 3)[pos]
        )
    ray_o_batch = torch.concat(ray_o_batch, dim=0).squeeze()  # [B*N_rays, 3]
    ray_d_batch = torch.concat(ray_d_batch, dim=0).squeeze()  # [B*N_rays, 3]
    mapped_img = torch.concat(mapped_img, dim=0).squeeze()  # [B*N_rays, 3]
    # print("batch", ray_o_batch.shape, ray_d_batch.shape, mapped_img.shape)
    return ray_o_batch, ray_d_batch, mapped_img


def c2w2Ray(c2w, sample_ray, params):
    height = params["height"]
    width = params["width"]
    focal = params["focal"]
    c2w = torch.Tensor(c2w)
    rays_o, rays_d, pos = GetRays(height, width, focal, c2w, sample_ray)
    return rays_o, rays_d, pos


def GetRays(height: int, width: int, focal: float, c2w, N_samples: int = None):
    """
    Derive the rays from camera

    :param height: height of the image
    :param width: width of the image
    :param focal: focal length of the camera
    :param c2w: transform matrix from camera coordinate to world coordinate
    :return:
        rays_d: directional vectors
        rays_o: starting points
        sampled_idx: indexes of sampled rays
    """
    idx_i, idx_j = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')

    idx_i = torch.flatten(idx_i)  # [H*W, 1]
    idx_j = torch.flatten(idx_j)  # [H*W, 1]

    if N_samples:
        sampled_idx = np.random.choice(np.arange(len(idx_i)), N_samples, replace=False)  # [N_rays, 1]
        idx_i = idx_i[sampled_idx][..., None]  # [N_rays, 1]
        idx_j = idx_j[sampled_idx][..., None]  # [N_rays, 1]
    else:
        sampled_idx = np.arange(len(idx_i))
        idx_i = idx_i[..., None]  # [N_rays, 1]
        idx_j = idx_j[..., None]  # [N_rays, 1]

    rays_d = torch.cat([(idx_i - width / 2) / focal,
                        -(idx_j - height / 2) / focal,
                        -torch.ones_like(idx_i)], dim=-1)  # [N_rays, 3]
    # print(rays_d.shape)
    rays_d = torch.sum(rays_d[..., None, :] * c2w[:3, :3], dim=-1)  # [N_rays, 3]
    rays_d = (rays_d / torch.norm(rays_d, dim=-1)[..., None]).squeeze()  # [N_rays, 3]
    # print(rays_d.shape)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)  # [N_rays, 3]

    return rays_o, rays_d, torch.tensor(sampled_idx, dtype=torch.int64)


def GetRays_np(height: int, width: int, focal: float, c2w, N_samples: int = None):
    """
    Derive the rays from camera

    :param height: height of the image
    :param width: width of the image
    :param focal: focal length of the camera
    :param c2w: transform matrix from camera coordinate to world coordinate
    :return:
        rays_d: directional vectors
        rays_o: starting points
        sampled_idx: indexes of sampled rays
    """
    idx_i, idx_j = np.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')

    idx_i = np.flatten(idx_i)  # [H*W, 1]
    idx_j = np.flatten(idx_j)  # [H*W, 1]

    if N_samples:
        sampled_idx = np.random.choice(np.arange(len(idx_i)), N_samples, replace=False)  # [N_rays, 1]
        idx_i = idx_i[sampled_idx][..., None]  # [N_rays, 1]
        idx_j = idx_j[sampled_idx][..., None]  # [N_rays, 1]
    else:
        sampled_idx = np.arange(len(idx_i))

    rays_d = np.stack([(idx_i - width / 2) / focal,
                       -(idx_j - height / 2) / focal,
                       -torch.ones_like(idx_i)], dim=-1)  # [N_rays, 3]
    rays_d = np.sum(rays_d[..., None, :] * c2w[:3, :3], dim=-1)  # [N_rays, 3]
    rays_d = (rays_d / np.norm(rays_d, dim=-1)[..., None]).squeeze()  # [N_rays, 3]
    # print(rays_d.shape)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)  # [N_rays, 3]

    return rays_o, rays_d, sampled_idx


def CreateModel(args):
    inPos_ch = args.inPos_ch
    inView_ch = args.inView_ch
    if args.embed_pos:
        inPos_ch = inPos_ch + inPos_ch * args.embed_pos * 2
    if args.embed_view:
        inView_ch = inView_ch + inView_ch * args.embed_view * 2

    model = nerf.NeRF(
        inPos_ch,
        inView_ch,
        args.pos_branch,
        args.out_ch,
        args.net_width,
        args.hidden_depth,
    )

    return model


def CreateOptimizer(args, models):
    params = []
    for model in models:
        params += model.parameters()

    optimizer = torch.optim.Adam(
        params,
        lr=args.lr,
        betas=(0.9, 0.999)
    )

    return optimizer
