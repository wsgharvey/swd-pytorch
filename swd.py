import numpy as np
import math
import torch
import torch.nn.functional as F


# Gaussian blur kernel
def get_gaussian_kernel(device="cpu"):
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]], np.float32) / 256.0
    gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
    return gaussian_k


def pyramid_down(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    # channel-wise conv(important)
    n_channels = image.shape[1]
    multiband = [F.conv2d(image[:, i:i + 1,:,:], gaussian_k, padding=2, stride=2)
                 for i in range(n_channels)]
    down_image = torch.cat(multiband, dim=1)
    return down_image


def pyramid_up(image, device="cpu"):
    gaussian_k = get_gaussian_kernel(device=device)
    upsample = F.interpolate(image, scale_factor=2)
    n_channels = image.shape[1]
    multiband = [F.conv2d(upsample[:, i:i + 1,:,:], gaussian_k, padding=2)
                 for i in range(n_channels)]
    up_image = torch.cat(multiband, dim=1)
    return up_image


def gaussian_pyramid(original, n_pyramids, device="cpu"):
    x = original
    # pyramid down
    pyramids = [original]
    for i in range(n_pyramids):
        x = pyramid_down(x, device=device)
        pyramids.append(x)
    return pyramids


def laplacian_pyramid(original, n_pyramids, device="cpu"):
    # create gaussian pyramid
    pyramids = gaussian_pyramid(original, n_pyramids, device=device)


    # pyramid up - diff
    laplacian = []
    for i in range(len(pyramids) - 1):
        diff = pyramids[i] - pyramid_up(pyramids[i + 1], device=device)
        laplacian.append(diff)
    # Add last gaussian pyramid
    laplacian.append(pyramids[len(pyramids) - 1])
    return laplacian


def minibatch_laplacian_pyramid(image, n_pyramids, batch_size, device="cpu"):
    n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
    pyramids = []
    for i in range(n):
        x = image[i * batch_size:(i + 1) * batch_size]
        p = laplacian_pyramid(x.to(device), n_pyramids, device=device)
        p = [x.cpu() for x in p]
        pyramids.append(p)
    del x
    result = []
    for i in range(n_pyramids + 1):
        x = []
        for j in range(n):
            x.append(pyramids[j][i])
        result.append(torch.cat(x, dim=0))
    return result


def extract_patches(pyramid_layer, slice_indices,
                    slice_size=7, unfold_batch_size=128, device="cpu"):
    assert pyramid_layer.ndim == 4
    n = pyramid_layer.size(0) // unfold_batch_size + np.sign(pyramid_layer.size(0) % unfold_batch_size)
    # random slice 7x7
    p_slice = []
    for i in range(n):
        # [unfold_batch_size, ch, n_slices, slice_size, slice_size]
        ind_start = i * unfold_batch_size
        ind_end = min((i + 1) * unfold_batch_size, pyramid_layer.size(0))
        x = pyramid_layer[ind_start:ind_end].unfold(
                2, slice_size, 1).unfold(3, slice_size, 1).reshape(
                ind_end - ind_start, pyramid_layer.size(1), -1, slice_size, slice_size)
        # [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
        x = x[:,:, slice_indices,:,:]
        # [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
        p_slice.append(x.permute([0, 2, 1, 3, 4]))
    # sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
    x = torch.cat(p_slice, dim=0)
    # reshape to 2rank
    n_channels = x.shape[-3]
    x = x.reshape(-1, n_channels * slice_size * slice_size)
    return x


class SWD():

    def __init__(self, C, H, W, n_pyramids=None, slice_size=7, n_descriptors=128,
                 n_repeat_projection=128, proj_per_repeat=4, device="cpu",
                 return_by_resolution=False, pyramid_batchsize=128,
                 smallest_res=16, seed=0):

        self.C = C
        self.H = H
        self.W = W
        if n_pyramids is None:
            n_pyramids = int(np.rint(np.log2(H // smallest_res)))
        self.n_pyramids = n_pyramids

        kwarg_names = ['n_pyramids', 'slice_size', 'n_descriptors',
                       'n_repeat_projection', 'proj_per_repeat', 'device',
                       'return_by_resolution', 'pyramid_batchsize',
                       'smallest_res', 'seed']
        for kwarg_name in kwarg_names:
            setattr(self, kwarg_name, eval(kwarg_name))

        self.sample_projections()
        self.all_projected1 = {i_pyramid: [] for i_pyramid in range(self.n_pyramids + 1)}   # will become n_images x n_descriptors x n_projections
        self.all_projected2 = {i_pyramid: [] for i_pyramid in range(self.n_pyramids + 1)}

    def sample_projections(self):

        prev_rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        self.projections = {}

        for i_pyramid in range(self.n_pyramids + 1):

            # indices
            H_i = int(self.H / 2**i_pyramid)
            W_i = int(self.W / 2**i_pyramid)
            n = (H_i-6) * (W_i-6)
            indices = torch.randperm(n)[:self.n_descriptors]

            n_projections = self.proj_per_repeat * self.n_repeat_projection
            projections = torch.randn(self.C*self.slice_size**2, n_projections).to(self.device)
            projections = projections / torch.std(projections, dim=0, keepdim=True)  # normalize

            self.projections[i_pyramid] = {'indices': indices,
                                           'projections': projections}

        torch.set_rng_state(prev_rng_state)

    def project_images(self, image1, image2):

        with torch.no_grad():
            # minibatch laplacian pyramid for cuda memory reasons
            pyramid1 = minibatch_laplacian_pyramid(image1, self.n_pyramids, self.pyramid_batchsize, device=self.device)
            pyramid2 = minibatch_laplacian_pyramid(image2, self.n_pyramids, self.pyramid_batchsize, device=self.device)
            result = []

            for i_pyramid in range(self.n_pyramids + 1):
                # indices
                indices = self.projections[i_pyramid]['indices']

                # extract patches on CPU
                # patch : 2rank (n_image*n_descriptors, slice_size**2*ch)
                p1 = extract_patches(pyramid1[i_pyramid], indices,
                                     slice_size=self.slice_size, device="cpu")
                p2 = extract_patches(pyramid2[i_pyramid], indices,
                                     slice_size=self.slice_size, device="cpu")

                p1, p2 = p1.to(self.device), p2.to(self.device)

                all_proj1 = []
                all_proj2 = []
                all_rand = self.projections[i_pyramid]['projections']
                for rand in torch.chunk(all_rand, chunks=self.n_repeat_projection, dim=1):
                    # rand = torch.randn(p1.size(1), proj_per_repeat).to(device)  # (slice_size**2*ch)
                    # rand = rand / torch.std(rand, dim=0, keepdim=True)  # normalize
                    # projection
                    proj1 = torch.matmul(p1, rand)
                    proj2 = torch.matmul(p2, rand)  # n_images*n_indices x proj_per_repeat
                    all_proj1.append(proj1)
                    all_proj2.append(proj2)

                self.all_projected1[i_pyramid].append(torch.cat(all_proj1, dim=1))
                self.all_projected2[i_pyramid].append(torch.cat(all_proj2, dim=1))

    def get_swd(self):

        sum_ = 0
        n_ = 0

        for i_pyramid in range(self.n_pyramids + 1):

            def distance(p1, p2):
                # compute summed over a minibatch of the descriptors
                if torch.cuda.is_available:
                    p1 = p1.cuda()
                    p2 = p2.cuda()
                p1, _ = torch.sort(p1, dim=0)
                p2, _ = torch.sort(p2, dim=0)
                return torch.sum(torch.abs(p1-p2)).cpu()

            proj1 = torch.cat(self.all_projected1[i_pyramid], dim=0)
            proj2 = torch.cat(self.all_projected2[i_pyramid], dim=0)

            n_chunks = math.ceil((proj1.numel()*4) / 100e6)  # roughly limit stuff moved to GPU to 100MB
            total_distance = 0
            for p1, p2 in zip(torch.chunk(proj1, chunks=n_chunks, dim=1),
                              torch.chunk(proj2, chunks=n_chunks, dim=1)):
                total_distance += distance(p1, p2) / proj1.numel()

            sum_ += total_distance
            n_ += 1

        return 1000 * sum_ / n_
