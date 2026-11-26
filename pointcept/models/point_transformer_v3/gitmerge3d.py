"""Train-free patch matching adapted from GitMerge3D's Sonata implementation."""

import torch
from torch.nn import functional as F


# Fixed-rate S3DIS patch preset in the GitMerge3D release.
PATCH_STRIDE = 5
MERGE_RATIO = 0.8


# TTR is temporarily disabled.
# def _ttr_values(destinations, sources, target):
    # """Fit one mean-centered ridge summary per matched V group, in FP32."""
    # channels = destinations.shape[-1]
    # bins = destinations.shape[2]
    # # Flatten patch and head into distinct group IDs.
    # offsets = torch.arange(
        # destinations.shape[0] * destinations.shape[1], device=target.device
    # ).reshape(*destinations.shape[:2], 1) * bins
    # group_ids = torch.cat(
        # (offsets + torch.arange(bins, device=target.device), offsets + target.squeeze(-1)),
        # dim=2,
    # ).flatten()
    # members = torch.cat((destinations, sources), dim=2).reshape(-1, channels)
    # order = group_ids.argsort()
    # result = destinations.reshape(-1, channels).clone()
    # counts = torch.bincount(group_ids, minlength=result.shape[0])
    # starts = counts.cumsum(0) - counts
#
    # # Batch equal-size groups without padding all groups to the largest one.
    # with torch.autocast(device_type=members.device.type, enabled=False):
        # for size in counts.unique().tolist():
            # if size == 1:
                # continue
            # groups = (counts == size).nonzero(as_tuple=True)[0]
            # positions = torch.arange(size, device=members.device)
            # # Solve in the smaller of token space and feature space.
            # dim = min(size, channels)
            # ridge = 0.1 * torch.eye(dim, device=members.device, dtype=torch.float32)
            # for batch in groups.split(max(1, 1048576 // (size * channels + dim * dim))):
                # indices = starts[batch, None] + positions
                # values = members[order[indices]].float()
                # keys = F.normalize(values, dim=-1)
                # mean = values.mean(dim=1).unsqueeze(-1)
                # residual = values.norm(dim=-1, keepdim=True) - keys @ mean
                # kt = keys.transpose(1, 2)
                # if size <= channels:
                    # correction = kt @ torch.linalg.solve(keys @ kt + ridge, residual)
                # else:
                    # correction = torch.linalg.solve(kt @ keys + ridge, kt @ residual)
                # result[batch] = (mean + correction).squeeze(-1).to(result.dtype)
    # return result.reshape_as(destinations)


def patch_matching(value, ratio=MERGE_RATIO, stride=PATCH_STRIDE):
    """Build per-head merge and unmerge operators for serialized patches."""
    batches, heads, tokens, _ = value.shape
    bins = tokens // stride

    def select(x, indices):
        return x.gather(2, indices.expand(-1, -1, -1, x.shape[-1]))

    with torch.no_grad():
        # One representative per full bin. Tail tokens are candidate sources.
        offsets = torch.randint(stride, (batches, heads, bins), device=value.device)
        destinations = offsets + stride * torch.arange(bins, device=value.device)
        flags = torch.zeros_like(value[..., 0], dtype=torch.long)
        flags.scatter_(2, destinations, -1)
        order = flags.argsort(dim=2).unsqueeze(-1)
        dst, src = order.split([bins, tokens - bins], dim=2)
        metric = F.normalize(value.float(), dim=-1)
        scores = select(metric, src) @ select(metric, dst).transpose(-2, -1)
        best, target = scores.max(dim=-1)
        ranked = best.argsort(dim=-1, descending=True).unsqueeze(-1)
        n_merge = min(int(tokens * ratio), tokens - bins)
        merged = select(src, ranked[:, :, :n_merge])
        kept = select(src, ranked[:, :, n_merge:])
        target = select(target.unsqueeze(-1), ranked[:, :, :n_merge])

    def merge(x, reduce="mean"):
        destinations, sources = select(x, dst), select(x, merged)
        # if reduce == "ttr":
        #     groups = _ttr_values(destinations, sources, target)
        groups = destinations.scatter_reduce(
            2, target.expand(-1, -1, -1, x.shape[-1]), sources, reduce
        )
        return torch.cat((select(x, kept), groups), dim=2)

    def unmerge(x):
        single, groups = x.split([kept.shape[2], bins], dim=2)
        out = x.new_empty(batches, heads, tokens, x.shape[-1])
        for indices, features in ((kept, single), (dst, groups), (merged, select(groups, target))):
            out.scatter_(2, indices.expand_as(features), features)
        return out

    return merge, unmerge
