import matplotlib.pyplot as plt
import torch
import nerf

"""
DataLoader
GetRays
Render
----CoarseSample
----GetRenderParams(coarse)
----FineSample
----GetRenderParams(fine)
----RenderRays
GradFeedback
"""


def draw_3d(ray):
    ax = plt.subplot(projection='3d')
    x, y, z = torch.split(ray, (1, 1, 1), dim=-1)
    x = x.detach().numpy()
    y = y.detach().numpy()
    z = z.detach().numpy()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.scatter(x, y, z)
    plt.show()
    plt.pause(0)


# def Render(ray_batch, renderCore, N_samples, device, base_samples=None, weights=None):
#     rays_o = ray_batch['rays_o'].to(device)
#     rays_d = ray_batch['rays_d'].to(device)
#     near = torch.Tensor([ray_batch['near']]).to(device)
#     far = torch.Tensor([ray_batch['far']]).to(device)
#     if base_samples is not None:
#         base_samples = base_samples.to(device)
#     if weights is not None:
#         weights = weights.to(device)
#
#     ray_pos, ray_view, seg_seq, z_samples = SampleRays(
#         rays_o,
#         rays_d,
#         near,
#         far,
#         N_samples,
#         device,
#         base_samples,
#         weights
#     )
#     vDense_seq, rgb_seq = GetRenderParams(ray_pos, ray_view, renderCore)
#     rgb_seq, weight_seq = RenderRays(seg_seq, vDense_seq, rgb_seq)
#     # print(rgb_seq.shape)
#     # assert rgb_seq.shape == (N_rays, N_samples, 3)
#     rgb = torch.sum(rgb_seq, dim=-2)
#     # if z_samples is not None:
#     #     print("z_samples ", z_samples.device)
#     # if weight_seq is not None:
#     #     print("weights ", weight_seq.device)
#     # if rgb is not None:
#     #     print("rgb ", rgb.device)
#     return {
#         "z_samples": z_samples,
#         "weights": weight_seq,
#         "rgb": rgb
#     }


def RenderRays(seg_seq, vDense_seq, rgb_seq):
    def AccTrans(dis, dense):
        return torch.exp(-torch.cumsum(dis * dense, dim=-1)) + 1e-10  # [N_rays, N_samples]

    def CurTrans(dis, dense):
        return 1. - torch.exp(-dis * dense) + 1e-10  # [N_rays, N_samples]

    # print(seg_seq.shape)
    trans_map = AccTrans(seg_seq, vDense_seq)  # [N_rays, N_samples]
    alpha_map = CurTrans(seg_seq, vDense_seq)  # [N_rays, N_samples]
    # print(trans_map.shape, alpha_map.shape)
    weight_seq = (trans_map * alpha_map)  # [N_rays, N_samples]
    rgb = torch.sum(weight_seq * rgb_seq, dim=-2)  # [N_rays, N_samples]
    rgb = torch.clip(
        rgb,
        torch.zeros_like(rgb),
        torch.ones_like(rgb)
    )
    return rgb, weight_seq


def Sample2Seg(z_samples, near, far):
    return torch.concat([torch.diff(z_samples * (far - near) + near),
                         torch.ones(list(z_samples.shape[:-1]) + [1]) * 1e7], dim=-1)[..., None]
    # [N_rays, N_samples]


def Sample2Ray(rays_o, rays_d, near, far, z_samples):
    # print(rays_d.device, rays_o.device, z_samples.device, far.device, near.device)
    return rays_o[..., None, :] + rays_d[..., None, :] * (z_samples[..., :, None] * (far - near) + near)
    # [N_rays, N_samples]


def GetNormalizedSamples(
        N_rays,
        N_samples,
        weights=None,
        coarse_samples=None,
        rand_sample=True
):
    if rand_sample:
        p_samples = torch.linspace(0, 1, N_samples + 1).expand(N_rays, N_samples + 1)  # [N_rays, N_samples+1]
        lowerb = p_samples[..., :-1]  # [N_rays, N_samples]
        upperb = p_samples[..., 1:]  # [N_rays, N_samples]
        p_samples = lowerb + torch.rand(lowerb.shape) * (upperb - lowerb)  # [N_rays, N_samples]
    else:
        p_samples = torch.linspace(0, 1, N_samples).expand(N_rays, N_samples)  # [N_rays, N_samples]

    if weights is None:
        return p_samples
    else:
        # print("weights ", weights.device)
        pdf = weights / torch.sum(weights[..., None, :], dim=-1)  # [N_rays, N_samples_coarse]
        # print("pdf ", pdf.device)
        cdf = torch.cumsum(pdf, dim=-1).squeeze(-1)  # [N_rays, N_samples_coarse]
        # print(cdf.device, p_samples.device)
        upper_bound = torch.searchsorted(cdf, p_samples, side='right')  # [N_rays, N_samples]
        rbound = torch.clamp(upper_bound, max=cdf.shape[-1] - 1)  # [N_rays, N_samples]
        lbound = torch.clamp(upper_bound - 1, min=0)  # [N_rays, N_samples]
        bounds = torch.concat([rbound[..., :, None], lbound[..., :, None]], dim=-1)  # [N_rays, N_samples,2]
        # print(coarse_samples.shape, bounds.shape)
        target_shape = [bounds.shape[0], bounds.shape[1], coarse_samples.shape[-1]]
        # print(target_shape)
        bound_z = torch.gather(coarse_samples[..., None, :].expand(target_shape), dim=-1,
                               index=bounds)  # [N_rays, N_samples, 2]
        bound_cdf = torch.gather(cdf[..., None, :].expand(target_shape), dim=-1,
                                 index=bounds)  # [N_rays, N_samples, 2]
        cdf_range = (bound_cdf[..., 1] - bound_cdf[..., 0])  # [N_rays, N_samples]
        z_range = (bound_z[..., 1] - bound_z[..., 0])  # [N_rays, N_samples]
        assert cdf_range.shape == (N_rays, N_samples)
        pctile = (p_samples - bound_cdf[..., 0]) / torch.where(cdf_range < 1e-5, torch.ones_like(cdf_range),
                                                               cdf_range)
        # [N_rays, N_samples]
        # print(coarse_samples.shape, (bound_z[..., 0] + z_range * pctile).shape)
        mix_samples = torch.concat([coarse_samples, bound_z[..., 0] + z_range * pctile], dim=-1)
        # [N_rays, N_fine + N_coarse]
        mix_samples, _ = torch.sort(mix_samples, dim=-1)  # [N_rays, N_fine + N_coarse]
        return mix_samples


def SampleRays(rays, N_samples, base_samples=None, weights=None, rand_sample=True):
    near = rays["near"]
    far = rays["far"]
    N_rays = len(rays["rays_o"])
    z_samples = GetNormalizedSamples(N_rays, N_samples, weights, base_samples)
    if weights is None:
        assert z_samples.shape == (N_rays, N_samples)
    else:
        assert z_samples.shape == (N_rays, N_samples + base_samples.shape[-1])
    # print(z_samples.shape)

    ray_pos = Sample2Ray(rays["rays_o"], rays["rays_d"], near, far,
                         z_samples)  # [N_rays, N_samples, 3]
    '''print(ray_pos.shape)
    if base_samples is None:
        draw_3d(ray_pos[0])
        exit(0)'''
    ray_view = torch.broadcast_to(rays["rays_d"][..., None, :], ray_pos.shape)  # [N_rays, N_samples, 3]
    # print(ray_pos.shape, ray_view.shape)
    if weights is None:
        assert ray_pos.shape == (N_rays, N_samples, 3)
    else:
        assert ray_pos.shape == (N_rays, N_samples + base_samples.shape[-1], 3)
    #

    seg_seq = Sample2Seg(z_samples, near, far)
    # ray_pos = ray_pos.to(device)
    # ray_view = ray_view.to(device)
    return ray_pos, ray_view, seg_seq, z_samples


class Renderer:
    def __init__(self, args, params):
        self.sample_coarse = args.sample_coarse
        self.sample_fine = args.sample_fine
        self.rand_sample = args.rand_sample
        self.ray_chunk = args.ray_chunk
        self.ray_batch = args.ray_batch
        self.renderCore = {}
        self.coarse_model = params["models"]["coarse"]
        self.fine_model = params["models"]["fine"]
        for tpe in ["coarse", "fine"]:
            core = lambda ray_pos, ray_view: nerf.RunNerF(
                ray_pos,
                ray_view,
                params["models"][tpe],
                params["embedders"]["pos"],
                params["embedders"]["view"],
            )
            if args.ray_chunk is None:
                self.renderCore[tpe] = core
            else:
                self.renderCore[tpe] = lambda ray_pos, ray_view: torch.cat(
                    [
                        core(
                            ray_pos[i: i + self.ray_chunk],
                            ray_view[i: i + self.ray_chunk]
                        ) for i in range(0, len(ray_pos), self.ray_chunk)
                    ],
                    dim=0
                )

    def BatchedRender(self, rays):
        if self.ray_batch is None:
            return self.Render(rays)
        else:
            stacked_res = {}
            for i in range(0, len(rays["rays_o"]), self.ray_batch):
                cur_res = self.Render(
                    {
                        "rays_o": rays["rays_o"][i: i + self.ray_batch],
                        "rays_d": rays["rays_d"][i: i + self.ray_batch],
                        "near": rays["near"],
                        "far": rays["far"],
                    }
                )
                for k in cur_res:
                    if k not in stacked_res:
                        stacked_res[k] = []
                    stacked_res[k].append(cur_res[k])
            for k in stacked_res:
                stacked_res[k] = torch.cat(stacked_res[k], dim=0)
            return stacked_res

    def GetRenderParams(self, rays_pos, rays_view, tpe=None):
        model_out = self.renderCore[tpe](rays_pos, rays_view)
        # print(model_out.shape)
        return torch.relu(model_out[..., :1]), torch.sigmoid(model_out[..., 1:])

    def Render(self, rays):
        ray_pos, ray_view, seg_seq, z_samples = SampleRays(
            rays,
            self.sample_coarse,
            rand_sample=self.rand_sample
        )
        # print("sample rays: ", toc - tic)
        vDense_seq, rgb_seq = self.GetRenderParams(ray_pos, ray_view, "coarse")
        # print("get params: ", toc - tic)
        rgb, weight_seq = RenderRays(seg_seq, vDense_seq, rgb_seq)
        # print("render rays: ", toc - tic)
        rgb_coarse = rgb
        ray_pos, ray_view, seg_seq, z_samples = SampleRays(
            rays,
            self.sample_fine,
            z_samples,
            weight_seq,
            rand_sample=self.rand_sample
        )
        vDense_seq, rgb_seq = self.GetRenderParams(ray_pos, ray_view, "fine")
        rgb, weight_seq = RenderRays(seg_seq, vDense_seq, rgb_seq)
        rgb_fine = rgb
        return {"coarse": rgb_coarse, "fine": rgb_fine}

    def __call__(self, rays):
        return self.BatchedRender(rays)

