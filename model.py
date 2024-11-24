from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn

from cellpose import models as cpmodels
from cellpose.resnet_torch import CPnet
from cellpose.core import assign_device


np.random.seed(13)
torch.manual_seed(13)
torch.cuda.manual_seed(13)

# important:
# to prevent mix-up between pytorch module eval and cellpose eval function
cpmodels.CellposeModel.evaluate = cpmodels.CellposeModel.eval


class SizeModelWrapper(cpmodels.SizeModel):
    """A wrapper around the cellpose SizeModel for estimating the cell diameter."""
    def __init__(self, cp_model, params):
        self.params = params
        self.diam_mean = params["diam_mean"]
        self.cp = cp_model


class CellPoseWrapper(nn.Module, cpmodels.CellposeModel):
    """
    A wrapper around the cellpose 'cyto3' model
    which is also act as a pytorch model.
    """
    def __init__(
        self, diam_mean=30., cp_batch_size=8, channels=[0, 0],
        flow_threshold=0.4, cellprob_threshold=0.0, stitch_threshold=0.0,
        estimate_diam=False, normalize=True, do_3D=False, gpu=True
    ):
        nn.Module.__init__(self)

        self.backbone = "default"
        self.diam_mean = diam_mean
        self.cp_batch_size = cp_batch_size
        self.channels = channels
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.stitch_threshold = stitch_threshold
        self.estimate_diam = estimate_diam
        self.normalize = normalize
        self.do_3D = do_3D
        self.nchan = 3
        self.nclasses = 3
        self.nbase = [2, 32, 64, 128, 256]
        self.mkl_enabled = torch.backends.mkldnn.is_available()
        self.diam_labels = diam_mean
        self.channel_axis = None
        self.invert = False
        self.gpu = gpu
        self.device = torch.device("cpu")
        self.device, self.gpu = assign_device(use_torch=True, gpu=gpu)

        self.net = CPnet(
            nbase=self.nbase, nout=self.nchan, sz=3,
            mkldnn=self.mkl_enabled, max_pool=True,
            diam_mean=self.diam_mean
        )

        # params for the size model from cyto3
        size_model_params = {}
        size_model_params["A"] = np.array([
            1.67477259e-01, -3.32731907e-01, -2.91088630e-01,  2.84643029e-01,
            1.35159547e+00, -3.36259193e-01, -6.52312327e-02,  4.03499655e-01,
            -4.86399971e-03, -1.47089789e-01,  2.29983242e-01,  1.25609383e+00,
            -5.62996904e-01, -1.50457766e-01, -6.11130023e-01, -3.86945777e-02,
            -1.15307716e-01, -6.18893911e-01,  7.49374837e-02, -4.39525736e-01,
            5.27069817e-01,  4.45777918e-02,  3.79584738e-01, -2.38056943e-01,
            3.54527124e-01, -3.74693772e-01,  5.61487649e-01,  1.22803735e-01,
            4.59874663e-01, -1.16211698e-01,  3.95158694e-01,  2.43836561e-01,
            1.68028918e-04, -2.55381289e-01,  2.28769092e-01,  1.07536833e-01,
            -5.38647559e-01, -3.36550544e-01,  4.87689538e-01,  2.68284740e-02,
            2.44259977e-01, -5.16894498e-01,  6.68113764e-01,  6.38549637e-01,
            -3.90951589e-01,  5.35833081e-01, -1.05660747e+00, -3.79715283e-01,
            -5.96134771e-03, -2.45152171e-01,  4.90848122e-01, -1.44375129e-01,
            -4.89655987e-01, -2.07489025e-01,  4.91282637e-01, -2.20302924e-01,
            8.52767211e-01, -6.95088104e-01, -3.74416238e-02,  6.62884046e-02,
            1.54268606e-02,  6.68190766e-02, -5.57925008e-01, -5.35231760e-01,
            -3.17334746e-01, -3.65853826e-01,  3.06920614e-02,  9.17921785e-01,
            -9.29930633e-01,  1.38519010e-01,  3.40381415e-01,  4.41879054e-01,
            6.27294434e-01,  1.19354061e-02, -1.72495874e-01,  1.99061651e-04,
            -3.19882822e-01,  2.96350014e-01,  3.28116235e-01, -3.25792502e-01,
            -6.90929781e-02, -5.41612283e-01,  6.17537564e-01, -2.46941766e-01,
            4.66782407e-01, -6.36419162e-01,  3.57788368e-01, -9.35420204e-02,
            1.05707089e+00, -3.82651565e-01, -1.97006285e-01, -3.94783797e-01,
            1.74262294e-01,  1.50808113e-03, -1.20036986e-01, -2.98115429e-01,
            -7.72391208e-02, -2.96818971e-01,  2.37227158e-01, -5.38003196e-01,
            1.33141701e+00,  2.02064562e-01, -2.39611478e-01,  3.44975250e-01,
            -7.79200546e-03, -1.65291121e-01, -5.40486299e-01,  8.73701830e-01,
            4.33780453e-01,  1.16539789e-01, -8.01846850e-01,  1.18619729e+00,
            2.96659895e-01,  8.31291214e-01, -6.68045928e-01,  2.58122716e-01,
            -3.82863425e-01, -3.86703591e-01,  3.46993556e-01,  2.79800368e-02,
            3.30047940e-01,  5.09909750e-01, -8.68419283e-01, -5.05084269e-01,
            -6.40695702e-01,  2.67957707e-01, -5.53567530e-01,  4.41581798e-01,
            1.88004902e-01,  1.98001919e-01,  2.16716691e-01,  1.44034023e-01,
            6.06299178e-01, -2.39383760e-01,  4.13124625e-01, -3.32679394e-01,
            7.79418726e-01, -2.86327977e-01, -6.54285269e-01, -1.50988843e-01,
            1.79699628e-01, -7.80435519e-01, -2.63631555e-01, -3.92324188e-01,
            1.47028773e+00,  2.70884957e-01, -3.95088638e-01,  5.09261973e-01,
            9.48169519e-02, -1.85574303e-01,  1.16995151e-01, -3.16811108e-01,
            6.07739592e-01,  3.13216090e-01,  8.92314078e-01,  6.11271308e-01,
            6.54678941e-01, -3.41591929e-01, -2.19198139e-01, -7.41658440e-01,
            -7.28836168e-01,  7.80775898e-01, -3.77374075e-01, -2.03348187e-01,
            -4.90522842e-01, -2.44295835e-01,  1.31084282e-01,  2.40628025e-03,
            -4.38873151e-01,  5.30493752e-02, -3.10191016e-01,  1.14249088e-01,
            -8.35475570e-02, -4.43363672e-01,  1.76076146e-01,  2.15535952e-01,
            3.00012749e-02, -2.37272124e-01,  7.67440706e-01, -8.81422544e-01,
            4.57145837e-01, -3.05151508e-01, -1.81967859e-01,  1.39191518e-01,
            2.62709313e-01,  6.50667080e-01, -2.79056529e-02,  8.21298394e-01,
            -8.27291112e-01, -9.14617971e-01, -2.19679694e-01, -4.83766149e-01,
            5.45524695e-01, -2.07467885e-01, -6.72863329e-01,  9.66901532e-02,
            -1.26351720e-02, -1.35390460e-01,  9.15522044e-04,  3.31833973e-01,
            5.69877577e-01, -3.64890154e-01, -1.11644380e+00,  1.72073151e-01,
            -6.40908959e-02, -1.06748430e-01,  5.08100539e-02,  6.65210826e-01,
            6.56440716e-02,  1.65875157e-01, -1.34274311e-01, -1.16138992e-02,
            -3.04370067e-01, -6.55805241e-01,  4.68139822e-02,  4.05333454e-01,
            -5.07461687e-01,  3.36557472e-01,  3.44882840e-01, -2.60437466e-01,
            -3.48844544e-01, -5.12006777e-02, -5.03223216e-01, -2.54741085e-01,
            -5.73661164e-01,  2.81261980e-02, -2.95974979e-02,  3.26348401e-01,
            -2.82012252e-01, -2.58912482e-01, -3.21288204e-02, -7.94997355e-02,
            -8.29782377e-02,  1.09676150e+00, -3.14985303e-01, -1.00603209e+00,
            -7.33197455e-01, -5.94386644e-01, -8.94490936e-01, -1.46844292e-01,
            -4.62075963e-01, -4.64324454e-01,  6.58523011e-02, -8.55183989e-01,
            -1.48004473e-01,  8.43811467e-01,  4.98597263e-01,  4.87130220e-01,
            5.57805850e-01,  4.85071676e-01,  2.03221836e-01, -4.60049283e-01,
            -4.25909565e-01, -1.25304865e+00, -1.38943136e-01, -6.41558003e-02
        ])
        size_model_params["smean"] = np.array([
            7.02817505e-03, -3.41898426e-02, -1.01446130e-06, -8.14546738e-03,
            -2.50744689e-02,  1.50047112e-02, -2.09066458e-02, -2.00457908e-02,
            8.05732980e-03, -2.21986473e-02, -1.16420193e-02, -1.80312037e-03,
            -8.50726757e-03, -3.15098688e-02,  2.21316013e-02,  3.07884589e-02,
            2.97576450e-02, -2.26491671e-02,  1.02468953e-02,  9.68619063e-03,
            3.32224779e-02, -1.66868567e-02, -2.80042104e-02,  2.57312991e-02,
            1.39168557e-02, -1.98610201e-02, -6.10362962e-02,  8.06582347e-03,
            1.15930811e-02,  2.07914952e-02, -1.73460562e-02,  2.05931850e-02,
            2.36991607e-03,  2.17302539e-03,  1.12199620e-03, -1.63450781e-02,
            3.78343684e-04,  1.47484606e-02,  1.76356360e-02, -6.11576019e-03,
            2.12664828e-02,  3.93447243e-02,  9.64876823e-03, -5.67259127e-03,
            1.84571184e-02, -2.73270924e-02,  7.44801341e-03, -2.08640881e-02,
            2.62937363e-04, -4.00068313e-02, -4.80764098e-02,  6.65812613e-03,
            -2.64079892e-03, -2.00415514e-02, -2.74631437e-02, -3.10419071e-02,
            2.43056603e-02,  4.65988740e-03,  1.52406413e-02,  2.24203104e-03,
            -1.02188466e-02, -2.22095614e-03,  2.76296902e-02, -1.21586248e-02,
            -1.83507726e-02,  3.51263657e-02,  2.68524215e-02, -2.31082187e-04,
            -1.13171749e-02, -3.00210677e-02,  4.25350899e-03,  1.77325122e-02,
            -4.47334303e-03,  3.69071327e-02,  8.10039882e-03, -1.77249387e-02,
            -9.05381702e-03, -1.48470411e-02, -2.91229021e-02,  4.89889458e-02,
            -1.07571008e-02,  2.58397553e-02,  2.53289472e-02, -2.32891515e-02,
            -1.62279271e-02,  5.96112087e-02,  4.32376638e-02,  2.36049537e-02,
            3.49053442e-02,  8.90851580e-03,  3.73389688e-03,  4.21875194e-02,
            3.10887210e-02, -4.17326093e-02, -1.89025514e-02, -3.96446884e-02,
            3.00624780e-02, -5.35325669e-02, -5.26981056e-03,  5.29729901e-03,
            2.85904594e-02, -1.21370414e-02, -2.79342867e-02, -3.13939042e-02,
            5.57508469e-02, -4.87103164e-02, -5.01185022e-02, -2.83574145e-02,
            6.64969608e-02,  2.72979531e-02, -2.25721151e-02,  2.35905834e-02,
            -1.00277271e-02, -1.74966850e-03,  2.43294369e-02,  2.34151147e-02,
            3.10633378e-03, -1.48890465e-02, -2.94535179e-02, -7.75304390e-03,
            -1.43072262e-04, -2.26182211e-02, -3.00467387e-02, -9.03409533e-03,
            7.72139383e-03, -3.71105876e-03, -4.65698540e-02,  4.32578474e-03,
            4.63743769e-02,  2.01347116e-02, -1.38433352e-02,  2.05824710e-02,
            -2.01259963e-02,  3.52871045e-02,  7.99413119e-03, -4.63261604e-02,
            1.82840843e-02,  1.36940405e-02,  3.04618310e-02,  4.27967869e-02,
            -1.22417044e-02,  2.28451062e-02,  7.12670013e-02, -2.78365687e-02,
            -4.39445563e-02,  2.11718846e-02,  4.66291048e-02,  3.05680255e-03,
            3.09582297e-02, -8.20493791e-03,  4.16333275e-03,  3.13657499e-03,
            2.43439735e-03, -9.20314901e-03, -8.36968943e-02,  3.14330123e-02,
            1.65023059e-02,  6.76610833e-03, -1.52854696e-02,  3.14306058e-02,
            -1.72477290e-02,  2.40754969e-02,  1.83787826e-03,  2.04304624e-02,
            -1.16792386e-02,  2.54101343e-02, -2.91021522e-02,  1.69194164e-03,
            1.85905844e-02,  2.11221147e-02, -2.13004798e-02, -2.62324908e-03,
            8.15639179e-03, -3.27417776e-02, -2.00890955e-02, -6.02603182e-02,
            -8.23608413e-03, -3.49385990e-03, -1.83386523e-02, -2.24783607e-02,
            1.29332058e-02, -4.25248966e-02,  1.36260083e-02, -5.99013604e-02,
            -2.24908609e-02, -1.10075762e-02,  4.09986936e-02, -4.19636853e-02,
            4.11421284e-02,  2.33651791e-02, -1.61375329e-02,  1.74532067e-02,
            -2.07987316e-02, -2.88488287e-02,  2.18136292e-02,  4.14803103e-02,
            -3.50480489e-02, -2.09283493e-02, -4.48528454e-02, -1.01049049e-02,
            -7.24977106e-02,  1.97470095e-02, -4.60452624e-02,  1.02777425e-02,
            5.18336473e-03, -3.54489237e-02, -3.00099552e-02,  1.08490055e-02,
            -1.01241386e-02,  1.87985301e-02, -6.95573539e-03, -6.83334190e-03,
            -5.72721660e-03, -2.19639912e-02,  4.62330282e-02,  1.27175348e-02,
            -3.88068780e-02,  8.01313762e-03,  7.34603871e-03, -4.61605424e-03,
            2.54217517e-02,  9.97360423e-03, -7.53617706e-03,  3.82113643e-02,
            -1.28267733e-02,  3.65535803e-02, -3.80081385e-02,  2.00994853e-02,
            6.28279429e-03,  8.08107108e-03, -2.66601071e-02,  4.60166298e-03,
            -1.17718512e-02,  3.15929689e-02,  5.12499288e-02,  2.34050746e-03,
            -7.40749948e-03, -3.43937352e-02,  2.86646895e-02, -1.33338105e-02,
            -5.46121970e-03, -9.68090538e-03, -5.92235029e-02, -3.75963002e-02,
            4.36179936e-02,  9.23077110e-03,  3.63224628e-03,  2.40146741e-02,
            -1.32300947e-02,  1.55432690e-02, -1.46348728e-02,  2.56410297e-02,
            -1.14090024e-02,  1.40860369e-02,  3.47977914e-02, -1.09329065e-02
        ])
        size_model_params["diam_mean"] = 30.
        size_model_params["ymean"] = -0.19780722
        self.size_model = SizeModelWrapper(self, size_model_params)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        matched = self.net.load_state_dict(state_dict, strict=strict, assign=assign)
        if len(matched.missing_keys) == 0 and len(matched.unexpected_keys) == 0:
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
            self.net.diameter_labels = self.diam_labels

        return matched

    def eval(self, *args, **kwargs):
        # pytorch module eval or cellpose model eval method?!
        if len(args) == 0 and len(kwargs) == 0:
            return self.train(False)
        else:
            return self.evaluate(*args, **kwargs)

    def forward(
        self, x
    ) -> Tuple[List[np.ndarray], List[List], List[np.ndarray], np.ndarray]:
        if len(x.shape) < 4:
            raise ValueError("input image(s) must be in 4-dimensional: b,c,y,x")

        # torch model input: b,c,y,x
        # cellpose input: list of numpy arrays in y,x,c
        image_list = [img.permute(1, 2, 0).cpu().numpy() for img in x]
        img_dims = len(image_list[0].shape)
        # estimating the diameter
        diams = self.diam_mean
        if self.estimate_diam and not self.do_3D and img_dims < 4:
            diams, _ = self.size_model.eval(
                image_list, channels=self.channels, channel_axis=None,
                batch_size=self.cp_batch_size, normalize=self.normalize,
                invert=False
            )
        # extracting masks
        masks_list, flows_list, style_list = self.eval(
            image_list, channels=self.channels,
            channel_axis=self.channel_axis,
            batch_size=self.cp_batch_size, normalize=self.normalize,
            invert=self.invert, diameter=diams, do_3D=self.do_3D,
        )

        # convert outputs to numpy arrays
        masks = np.array(masks_list, dtype=np.float32)
        styles = np.array(style_list, dtype=np.float32)
        # flows: stack them together
        # TODO: each image flow can be a list of 3 or 4. but here we ignore the 4th element anyway.
        flows = []
        for fl in flows_list:
            f_arr = np.vstack([
                np.moveaxis(fl[0], 2, 0),
                fl[1],
                fl[2][np.newaxis],
            ], dtype=np.float32)
            flows.append(f_arr)
        flows = np.array(flows)

        # add batch dim to dims
        if self.estimate_diam:
            diams = np.array(np.round(diams, 5)).reshape(-1, 1)
        else:
            diams = np.array(np.round(diams, 5)).repeat(x.shape[0]).reshape(-1, 1)
        assert diams.shape[0] == x.shape[0]

        return masks, flows, styles, diams


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tifffile

    tiff_images = tifffile.imread("./data/test_images.tif")
    print(tiff_images.shape)
    img_batch = torch.from_numpy(tiff_images).unsqueeze(1)
    print(img_batch.shape) # should be b,c,y,x

    model = CellPoseWrapper(estimate_diam=True, gpu=False)
    model.load_state_dict(
        torch.load("../original/cellpose_models/cyto3", map_location=model.device)
    )
    # torch.save(model.state_dict(), "./model_weights.pth")

    masks, flows, styles, diams = model(img_batch)
    print(masks.shape, flows.shape, styles.shape, diams)

    fig, axes = plt.subplots(1, len(masks), figsize=(4 * len(masks), 7))
    for i in range(len(masks)):
        axes[i].imshow(masks[i], cmap="Set2")
    plt.show()
