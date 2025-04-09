from typing import Tuple, List
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from cellpose import models as cpmodels
from cellpose.resnet_torch import CPnet
from cellpose.core import assign_device


np.random.seed(13)
torch.manual_seed(13)
torch.cuda.manual_seed(13)

# !important:
# to prevent mix-up between pytorch module eval and cellpose eval functions
cpmodels.CellposeModel.evaluate = cpmodels.CellposeModel.eval


class SizeModelWrapper(cpmodels.SizeModel):
    """A wrapper around the cellpose SizeModel for estimating the cell diameter."""
    def __init__(self, cp_model, params):
        self.params = params
        self.diam_mean = params["diam_mean"]
        self.cp = cp_model


class CellPoseWrapper(nn.Module, cpmodels.CellposeModel):
    """
    A wrapper around the cellpose model
    which is also act as a pytorch model.
    """
    def __init__(
        self, model_type="cyto3", diam_mean=None, cp_batch_size=8, channels=[0, 0],
        flow_threshold=0.4, cellprob_threshold=0.0, stitch_threshold=0.0,
        estimate_diam=False, normalize=True, do_3D=False, gpu=True
    ):
        nn.Module.__init__(self)

        self.backbone = "default"
        self.model_type = model_type
        self.diam_mean = diam_mean
        if self.diam_mean is None:
            if self.model_type == "nuclei":
                self.diam_mean = 17.
            else:
                self.diam_mean = 30.
        self.cp_batch_size = cp_batch_size
        self.channels = channels
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.stitch_threshold = stitch_threshold
        self.estimate_diam = estimate_diam
        self.normalize = normalize
        self.do_3D = do_3D
        self.nchan = 2
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
            nbase=self.nbase, nout=self.nclasses, sz=3,
            mkldnn=self.mkl_enabled, max_pool=True,
            diam_mean=self.diam_mean
        )

        # params for the size model
        size_model_params = {}
        if self.model_type == "nuclei":
            size_model_params["A"] = np.array([
                -7.30257322e-03, -1.18541566e-01, -1.45443049e+00,  8.12052821e-02,
                -3.36623138e-01,  5.69662230e-01,  3.11142688e-01, -6.09824024e-01,
                -6.43741585e-01, -5.98831833e-01, -4.81707675e-01, -3.36116706e-02,
                6.28041388e-01,  4.23793396e-01, -1.12342098e-01, -8.13136855e-01,
                -4.19454064e-01, -1.02335098e-01, -5.25129934e-02, -7.48914524e-01,
                1.98812840e-01,  7.10478179e-03,  1.71442370e-01, -7.77180606e-01,
                3.80796811e-01,  5.11640578e-02, -4.26206144e-02, -3.42497682e-01,
                -4.61923956e-01,  5.66295755e-02, -6.98916768e-02,  1.14903573e+00,
                6.14386375e-02,  3.23539467e-01,  6.68056423e-01, -3.82598754e-01,
                9.19155554e-02, -5.75596295e-01, -2.89050530e-01, -3.73992147e-02,
                -4.06119627e-01,  4.39080641e-01, -6.76666708e-03,  6.87245177e-02,
                -9.33877502e-02, -8.79418552e-01,  1.91468115e-01,  9.36622324e-01,
                1.01726749e-02,  2.44981405e-01, -6.80783885e-01, -6.39704188e-02,
                1.28580357e-01, -1.15498299e-01, -5.71837878e-01,  1.93789015e-01,
                -1.24737299e-01, -2.69296561e-01,  3.49892227e-01,  2.79277008e-01,
                2.73074038e-01,  1.01448370e+00,  6.50989370e-01,  4.54412708e-01,
                -5.01466865e-01, -5.47891959e-01,  1.44800898e-01, -8.14151325e-02,
                2.66403259e-01, -1.13082328e-02,  6.67414612e-01,  1.60694881e-01,
                -3.64724210e-01, -3.18854758e-01, -2.47218132e-01,  3.74598889e-01,
                2.22578597e-01,  2.64687052e-01,  5.21234271e-01,  2.28599445e-01,
                -1.36529850e-01, -1.09733859e-01, -1.29851097e+00,  9.54783752e-02,
                -4.10788447e-01, -5.41957821e-02,  3.85910082e-01, -4.06507879e-01,
                -7.21773282e-01, -4.28815918e-01,  6.76072962e-01,  1.75512178e-01,
                3.24537319e-01, -1.71334451e-01,  2.72745316e-01, -4.41583487e-01,
                -4.55115198e-02, -3.97578122e-01, -9.24241524e-02, -7.54304872e-01,
                -2.94896281e-01,  2.46921771e-01, -6.17179450e-01, -8.80150329e-01,
                1.50621315e-01,  5.01621143e-01,  3.85757371e-01,  2.36261854e-01,
                -4.18688351e-01,  1.61476140e-01,  7.88219006e-01, -2.73492022e-01,
                -7.19490666e-01,  1.29134377e-01,  1.40948291e-01,  5.73052406e-01,
                -1.50786369e-01, -4.26432995e-01,  1.19872624e-01,  3.60309361e-01,
                -4.71223496e-01, -1.68708617e-01, -4.80614652e-01, -3.75616766e-02,
                6.02355950e-01,  4.98677369e-01,  3.50356698e-01,  1.27891399e+00,
                -7.34913090e-01,  4.59687623e-01,  5.07250940e-01,  2.06066144e-01,
                3.40622481e-01,  8.66120088e-02,  3.79763322e-01,  6.06499243e-04,
                4.57963420e-01, -1.87068105e-01, -2.02574417e-01, -7.20818741e-01,
                8.40157187e-01, -1.18396872e-01, -4.28434101e-01,  3.34318694e-01,
                5.91669224e-01, -2.27607682e-01, -4.25011564e-01, -5.63305651e-01,
                -4.37746596e-01,  1.25207809e+00, -1.92788323e-01,  8.40720758e-01,
                -2.52284402e-01,  3.88115320e-01,  4.29404700e-01, -7.24217943e-01,
                -2.77653334e-03,  1.21906516e+00,  4.60775968e-01,  4.89036997e-02,
                5.62757548e-01,  2.08859916e-01, -7.96334555e-01, -7.24607802e-01,
                7.88552387e-02, -1.02246815e+00,  9.25539605e-01, -1.66869343e-01,
                -2.61723160e-01, -2.87214532e-01,  5.29071590e-01,  3.41658246e-01,
                -2.65158603e-01, -1.38325040e-02, -4.80279877e-01, -3.92908220e-01,
                5.04147353e-01,  2.64595905e-01,  1.92707480e-01, -4.74048452e-01,
                -2.76101528e-01,  5.95012295e-01, -5.57747099e-01,  3.29355864e-01,
                -3.01069223e-01,  8.90298996e-02,  3.00777855e-01,  4.38696701e-01,
                5.74469666e-02, -5.19230426e-01,  1.12916631e+00,  1.71083862e-01,
                7.60297492e-02,  4.36458929e-01, -2.35838602e-01, -5.65566325e-02,
                -2.65516149e-03, -7.46021118e-02,  1.81973536e-01, -6.55610977e-01,
                -7.03301365e-01,  8.96824268e-01, -1.39412224e-01, -8.90077096e-01,
                -1.17441535e+00,  1.12941105e-02, -1.14367977e+00, -4.04694058e-01,
                -1.10952721e-01,  1.38066794e-01, -3.37494400e-01,  1.18612870e+00,
                5.22387550e-01, -3.71397238e-01,  1.73067594e-01,  2.33206907e-01,
                -8.19779233e-01, -4.78900567e-02,  1.03033526e+00, -1.17887950e-01,
                -2.83783118e-01, -5.04461106e-01,  1.52220128e-01, -1.43789853e-01,
                -7.82530470e-01, -6.48564625e-01, -3.20363939e-01,  2.58724496e-01,
                -3.45678124e-01,  2.02746794e-01, -4.37875355e-01, -8.53299582e-02,
                3.10806509e-01, -8.21932216e-02, -4.67703094e-02, -4.45931215e-01,
                -8.38303638e-01,  5.01171273e-01,  1.03974207e-01, -1.20771798e-01,
                -2.73155046e-01, -1.36293463e-01,  7.53784352e-01, -3.16267625e-01,
                -2.13172267e-01, -3.13186275e-01,  1.54900109e-01,  1.16005650e+00,
                -1.41505965e-01,  4.95146120e-01, -3.55510702e-01, -5.47924051e-03,
                3.34128465e-01,  4.85135112e-01,  3.98947991e-01, -4.07170559e-01
            ])
            size_model_params["smean"] = np.array([
                3.33571546e-02, -1.33421663e-02, -7.72377774e-02, -2.62366012e-02,
                6.51190281e-02,  2.34221667e-02,  5.43226749e-02,  3.60966511e-02,
                7.01446533e-02,  1.16591118e-01,  6.58042962e-03, -2.78950408e-02,
                7.46760983e-03,  5.05509786e-02,  4.77152765e-02,  1.93309244e-02,
                2.15454157e-02,  1.88199189e-02,  5.67697100e-02,  5.54717407e-02,
                2.36356948e-02,  1.31597025e-02,  4.51800190e-02, -2.14927201e-03,
                1.64783839e-02,  1.32892681e-02, -6.69834688e-02,  3.13859656e-02,
                -1.36573408e-02,  5.19549884e-02,  6.48780838e-02, -8.45314120e-04,
                -2.78426223e-02,  5.04880846e-02,  2.22085975e-02,  4.96192090e-02,
                -2.25106422e-02,  3.30135524e-02,  3.22429650e-03,  5.05473688e-02,
                -2.14093681e-02,  5.58082713e-03,  3.18176895e-02,  2.44566165e-02,
                -1.97110530e-02, -1.57422815e-02,  2.29600854e-02,  3.86349708e-02,
                -3.74921486e-02, -8.73819143e-02,  4.75215167e-03,  4.21954095e-02,
                1.21927075e-02,  2.61907256e-03,  1.10887326e-01, -3.32249142e-02,
                -4.06566896e-02,  1.07597345e-02, -7.32940435e-03, -4.49081771e-02,
                -4.77932170e-02, -1.42577710e-02,  2.37081721e-02,  7.10527524e-02,
                -1.71866864e-02,  5.86929955e-02, -8.10543355e-03,  4.70209634e-03,
                2.06222348e-02,  3.97141613e-02,  1.41979929e-03, -3.72401737e-02,
                1.36016281e-02, -1.93084069e-02, -1.20762050e-01,  8.43954459e-02,
                6.36447072e-02,  2.04727869e-03,  3.80113721e-02,  7.42430463e-02,
                3.34038702e-03,  7.07524493e-02, -5.37632182e-02,  1.76843077e-01,
                3.10399197e-02, -3.97474356e-02,  6.22577928e-02, -3.63351703e-02,
                5.71441799e-02, -1.72642022e-02,  1.92099321e-03,  4.62237261e-02,
                -1.66606286e-03,  4.03431766e-02,  4.70224675e-03, -3.74267995e-03,
                -1.39494683e-03,  5.16524576e-02,  6.70298785e-02, -6.83201030e-02,
                2.99535622e-03,  3.06514557e-02, -2.09709629e-02, -2.25120820e-02,
                6.90594502e-03, -6.28922209e-02,  5.31853996e-02,  7.06917495e-02,
                -7.50806034e-02,  4.02728580e-02,  4.61776294e-02, -3.40574495e-02,
                -4.66963202e-02,  3.26143727e-02, -2.23153830e-02,  3.01710772e-03,
                -2.18282212e-02, -1.91387814e-02,  1.72387678e-02,  1.53471068e-01,
                7.81310797e-02, -2.88260058e-02, -1.75903682e-02,  2.84443833e-02,
                6.28208593e-02, -7.38600036e-03,  4.81804684e-02, -5.43766655e-03,
                4.84366454e-02,  2.09212769e-02,  5.36952578e-02, -2.89113056e-02,
                9.52697843e-02,  5.93781397e-02,  5.28115220e-02,  1.92410150e-03,
                3.57586481e-02, -1.66731315e-05, -6.59857690e-02, -1.13312736e-01,
                2.20412835e-02,  7.92401955e-02, -1.84811223e-02,  8.68499745e-03,
                3.64950113e-02,  2.66536623e-02,  2.18437407e-02, -2.28698701e-02,
                9.19985771e-03, -5.09864241e-02,  4.80524153e-02,  2.20944788e-02,
                7.03299567e-02,  1.06560858e-02, -1.30806433e-03,  1.96260903e-02,
                5.87253198e-02,  1.09164946e-01,  4.92166243e-02,  1.07307605e-01,
                1.74453110e-02,  4.30702232e-02, -1.54439984e-02,  7.49207586e-02,
                5.07971533e-02,  3.72537831e-03, -3.34328637e-02, -6.09636605e-02,
                -4.85441508e-03,  6.03521802e-02,  3.51783372e-02,  7.83585832e-02,
                5.00762388e-02, -2.40870286e-02,  2.03905404e-02,  4.59519997e-02,
                1.68489262e-01,  6.22320212e-02,  9.82332081e-02, -8.62647593e-03,
                8.92681070e-03,  4.42702770e-02,  2.48581339e-02, -6.89907232e-03,
                3.88028473e-02, -1.93572305e-02,  2.05844473e-02, -5.50207086e-02,
                -5.21811768e-02,  4.36080061e-02, -5.50148226e-02, -2.36560628e-02,
                1.44468054e-01,  2.74861082e-02,  6.03093475e-04, -2.22522821e-02,
                3.15281339e-02,  2.49214401e-03,  4.04867018e-03, -1.99645963e-02,
                -6.54805498e-03,  5.53097948e-02,  3.80778387e-02,  4.01360802e-02,
                4.17350903e-02,  6.53876550e-03, -2.44954433e-02,  2.85442378e-02,
                1.72955450e-02,  6.97525069e-02, -1.83215085e-02, -8.81236717e-02,
                6.60927296e-02,  6.08626846e-03,  2.82136090e-02,  5.34143709e-02,
                -9.36697647e-02,  4.38540801e-02,  5.05244248e-02,  4.63728458e-02,
                3.54122929e-03,  3.79999541e-02,  4.38324288e-02, -5.40515175e-03,
                -9.27755609e-03,  9.94902849e-03,  5.62093295e-02,  3.93828526e-02,
                9.98743158e-03, -3.16818915e-02,  1.03675965e-02,  4.55703922e-02,
                1.57460067e-02,  1.87545214e-02, -1.66561734e-02,  7.04655126e-02,
                -4.12616786e-03,  1.40790403e-01, -6.62945863e-03,  3.59633416e-02,
                6.70748577e-02,  1.03649184e-01,  1.28174364e-03,  7.59943575e-02,
                9.04740319e-02,  6.08805865e-02,  7.32386559e-02,  7.99016282e-02,
                -7.85914063e-02,  7.44997710e-02, -7.96964206e-03,  2.22319383e-02,
                -8.92254338e-03,  1.15279062e-02, -3.47399339e-03,  1.18272686e-02
            ])
            size_model_params["diam_mean"] = 17.
            size_model_params["ymean"] = 0.1394341
        else:
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
        assert state_dict["output.2.weight"].shape[0], self.net.nout
        Incompatible = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
        result = Incompatible([], [])

        if state_dict["output.2.weight"].shape[0] != self.net.nout:
            for name in self.net.state_dict():
                if "output" not in name:
                    self.net.state_dict()[name].copy_(state_dict[name])
        else:
            result = self.net.load_state_dict(
                dict([(name, param) for name, param in state_dict.items()]),
                strict=False)

        self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]      # ROIs rescaled to this size during training
        self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]  # mean diameter of training ROIs

        return result

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
        diams = self.diam_labels  # diameter used for training / fine-tuning
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
            diameter=diams,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            stitch_threshold=self.stitch_threshold,
            batch_size=self.cp_batch_size, normalize=self.normalize,
            invert=self.invert, do_3D=self.do_3D,
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
    img_batch = torch.from_numpy(tiff_images).unsqueeze(1).permute(0, 2, 3, 1)
    print(img_batch.shape) # should be channel last

    model = CellPoseWrapper(estimate_diam=True)
    model.load_state_dict(
        torch.load("./cellpose_models/cyto3", map_location=model.device)
    )
    # torch.save(model.state_dict(), "./model_weights.pth")

    masks, flows, styles, diams = model(img_batch)
    print(masks.shape, flows.shape, styles.shape, diams)

    fig, axes = plt.subplots(1, len(masks), figsize=(4 * len(masks), 7))
    for i in range(len(masks)):
        axes[i].imshow(masks[i], cmap="Set2")
    plt.show()
