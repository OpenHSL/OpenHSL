from matplotlib import pyplot as plt

from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask, draw_colored_mask

from openhsl.nn.data.utils import HyperMinMaxScaler, HyperStandardScaler, apply_pca
from openhsl.nn.models.m1dcnn import M1DCNN
from openhsl.nn.models.tf2dcnn import TF2DCNN
from openhsl.nn.models.m3dcnn_li import M3DCNN
from openhsl.nn.models.nm3dcnn import NM3DCNN
from openhsl.nn.models.ssftt import SSFTT
from openhsl.nn.models.utils import draw_fit_plots, get_accuracy, get_f1

HSI_PATH = './test_data/tr_pr/PaviaU.mat'
HSI_KEY = 'paviaU'
MASK_PATH = './test_data/tr_pr/PaviaU_gt.mat'
MASK_KEY = 'paviaU_gt'

SCALER_TYPE = 'Standard'
SCALE_PER = 'band'

USE_PCA = False
NUM_COMPONENTS = 30

CLASSIFIER = 'M1DCNN'

USE_WANDB = False

OPTIM_LEARNING_RATE = 0.01
OPTIM_WEIGHT_DECAY = 0

SCHEDULER_TYPE = 'StepLR'
SCHED_STEP_SIZE = 5
SCHED_GAMMA = 0.5

EPOCHS = 50
TRAIN_SAMPLE_PERCENTAGE = 0.1
DATALOADER_MODE = 'fixed'
BATCH_SIZE = 32

f1_type = 'weighted'

hsi = HSImage()
mask = HSMask()

hsi.load(path_to_data=HSI_PATH,
         key=HSI_KEY)
mask.load(path_to_data=MASK_PATH,
          key=MASK_KEY)

if SCALER_TYPE == 'Standard':
    scaler = HyperStandardScaler(per=SCALE_PER)
elif SCALER_TYPE == 'MinMax':
    scaler = HyperMinMaxScaler(per=SCALE_PER)
else:
    raise ValueError('Unsupported scaler type')

hsi.data = scaler.fit_transform(hsi.data)

if USE_PCA:
    hsi.data, _ = apply_pca(hsi.data, NUM_COMPONENTS)

CLASSIFIERS = {
    "M1DCNN": M1DCNN,
    "TF2DCNN": TF2DCNN,
    "M3DCNN": M3DCNN,
    "NM3DCNN": NM3DCNN,
    "SSFTT": SSFTT
}


_CLASSIFIER = CLASSIFIERS[CLASSIFIER]

optimizer_params = {
    "learning_rate": OPTIM_LEARNING_RATE,
    "weight_decay": OPTIM_WEIGHT_DECAY
}

scheduler_params = {
    "step_size": SCHED_STEP_SIZE,
    "gamma": SCHED_GAMMA
}

fit_params = {
    "epochs": EPOCHS,
    "train_sample_percentage": TRAIN_SAMPLE_PERCENTAGE,
    "dataloader_mode": DATALOADER_MODE,
    "optimizer_params": optimizer_params,
    "batch_size": BATCH_SIZE,
    "scheduler_type": SCHEDULER_TYPE,
    "scheduler_params": scheduler_params
}

net = _CLASSIFIER(n_classes=mask.n_classes,
                  n_bands=hsi.data.shape[-1],  # or hsi.data.shape[-1]
                  device='cuda')

if USE_WANDB:
    net.init_wandb()

net.fit(X=hsi,  # or hsi
        y=mask.get_2d(),
        fit_params=fit_params)

draw_fit_plots(model=net)

pred = net.predict(X=hsi,  # or hsi
                   y=mask,
                   batch_size=100)

pred = pred * (mask.get_2d() > 0)

plt.imshow(pred)
plt.show()

draw_colored_mask(mask=mask,
                  predicted_mask=pred,
                  stack_type='h')

acc = get_accuracy(mask.get_2d(), pred)
f1 = get_f1(mask.get_2d(), pred, average=f1_type)

print(f'Accuracy: {acc}')
print(f'F1 {f1_type}: {f1}')
