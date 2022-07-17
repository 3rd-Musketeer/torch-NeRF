import torch
import torch.nn as nn


class Embedder:
    def __init__(self, L, fn_list=None):
        if fn_list is None:
            fn_list = [torch.sin, torch.cos]
        buk = [2 ** i for i in range(L)]
        buk = torch.Tensor(buk)
        self.outFn = [lambda x: x]
        for coef in buk:
            for fn in fn_list:
                self.outFn.append(lambda x, coef=coef, fn=fn: fn(x * coef))

    def __call__(self, in_vec):
        return torch.concat([fn(in_vec) for fn in self.outFn], dim=-1)


class NeRF(nn.Module):
    def __init__(
            self,
            inPos_ch,
            inView_ch,
            pos_branch,
            out_ch,
            net_width,
            hidden_depth,
    ):
        super(NeRF, self).__init__()
        self.pos_branch = pos_branch
        self.relu = nn.ReLU(inplace=True)

        pos_layer = [nn.Linear(inPos_ch, net_width)] + [
            nn.Linear(net_width, net_width) if i not in pos_branch
            else nn.Linear(net_width + inPos_ch, net_width) for i in range(hidden_depth - 1)
        ]

        self.pos_network = nn.ModuleList(pos_layer)

        self.view_network = nn.Linear(net_width + inView_ch, net_width // 2)

        self.output = nn.Linear(net_width // 2, out_ch)

    def forward(self, rays_pos, rays_view):
        x = rays_pos

        for i, layer in enumerate(self.pos_network):
            # print(i, x.shape)
            if i - 1 in self.pos_branch:
                x = torch.concat([x, rays_pos], dim=-1)
            x = self.relu(layer(x))

        x = torch.concat([x, rays_view], dim=-1)
        x = self.view_network(x)

        x = self.output(x)

        return x


def RunNerF(rays_pos, rays_view, model, embed_pos=None, embed_view=None):
    if embed_pos is not None:
        rays_pos = embed_pos(rays_pos)
    if embed_view is not None:
        rays_view = embed_view(rays_view)
    return model(rays_pos, rays_view)
