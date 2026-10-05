from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import numpy as np
import torch

def compute_ssim_from_model(model, dataloader, device):
    model.eval()
    ssim_scores = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)  # assuming (x, y) pairs
            _, outputs = model(inputs)

            inputs_np = inputs.cpu().numpy()
            outputs_np = outputs.cpu().numpy()

            # Assume inputs are in [0,1] and shape [B, C, H, W]
            for i in range(inputs_np.shape[0]):
                inp = np.transpose(inputs_np[i], (1, 2, 0))  # to HWC
                out = np.transpose(outputs_np[i], (1, 2, 0))
                
                # If grayscale, remove channel dimension
                if inp.shape[2] == 1:
                    inp = inp.squeeze(-1)
                    out = out.squeeze(-1)
                
                ssim_val = ssim(inp, out, data_range=1.0, channel_axis=None if inp.ndim == 2 else -1)
                ssim_scores.append(ssim_val)

    return ssim_scores