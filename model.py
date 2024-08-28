import numpy as np
import torch
import torch.nn as nn

from cellpose import models


class CellPoseWrapper(nn.Module):
    """A delegate model which runs the cellpose over the input images."""
    def __init__(
        self, model_type="cyto", channels=[0, 0], diameter=None,
        batch_size=8, do_3D=False
    ):
        super().__init__()
        # check for GPU
        on_gpu = torch.cuda.is_available()
        # define the actual cellpose model
        self.model_type = model_type
        self.channels = channels
        self.diameter = diameter
        self.batch_size = batch_size
        self.do_3D = do_3D
        # initialize cellpose
        self.cellpose = models.Cellpose(
            gpu=on_gpu,
            model_type=model_type
        )
        # just to have some weights ;)
        self.fc = nn.Linear(1, 1)
        self.eval()

    def forward(self, x):
        # torch model input: b,c,y,x
        # cellpose input: list of numpy arrays in y,x,c
        images = [img.permute(1, 2, 0).cpu().numpy() for img in x]
        masks_list, flows_list, styles, diams = self.cellpose.eval(
            images, batch_size=self.batch_size,
            channels=self.channels, diameter=None,
            do_3D=self.do_3D
        )
        # convert outputs to numpy arrays
        # masks
        masks = np.array(masks_list, dtype=np.float32)
        # flows
        flows_arr = []
        for flows in flows_list:
            f_arr = np.vstack([
                np.moveaxis(flows[0], 2, 0),
                flows[1],
                flows[2][np.newaxis],
                flows[3]
            ], dtype=np.float32)
            flows_arr.append(f_arr)
        flows_arr = np.array(flows_arr)

        return masks, flows_arr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tifffile

    tiff_images = tifffile.imread("./test_images.tif")
    print(tiff_images.shape)
    img_batch = torch.from_numpy(tiff_images).unsqueeze(1)
    print(img_batch.shape)

    model = CellPoseWrapper()
    torch.save(model.state_dict(), "./model_weights.pth")

    masks, flows = model(img_batch)
    print(masks.shape, flows.shape)

    fig, axes = plt.subplots(1, len(masks), figsize=(4 * len(masks), 7))
    for i in range(len(masks)):
        axes[i].imshow(masks[i], cmap="Set2")
    plt.show()
