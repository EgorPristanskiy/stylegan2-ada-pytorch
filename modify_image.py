# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import re
import numpy as np
from typing import List, Optional

import click
import imageio
import dnnlib
import PIL.Image
import torch
from modification.latent_space_modifier import LatentSpaceModifier

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE', required=True)
@click.option('--lowest-value-w', help='Projection result file with lowest value', type=str, metavar='FILE', required=True)
@click.option('--highest-value-w', help='Projection result file with highest value', type=str, metavar='FILE', required=True)
@click.option('--out-video', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    noise_mode: str,
    out_video: str,
    projected_w: Optional[str],
    lowest_value_w: Optional[str],
    highest_value_w: Optional[str]
):
    modification_shift = 0.005

    print('Loading networks from "%s"...' % network_pkl)
    modifier = LatentSpaceModifier(
        lowest_value_w, highest_value_w, projected_w, modification_shift=modification_shift
    )
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if (device.type == 'cpu'):
        G = G.float()

    video = imageio.get_writer(out_video, mode='I', fps=10, codec='libx264', bitrate='16M')
    for _ in np.arange(0.0, 0.2, modification_shift):
        ws = modifier.move_image()
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for _, w in enumerate(ws):
            if (device.type == 'cpu'):
                synth_image = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode, force_fp32=True)
            else:
                synth_image = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)

            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(synth_image)
    video.close()
    return


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
