import torch


def compute_locations(features, stride, offset):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride, offset,
        features.device
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, offset, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + offset
    return locations