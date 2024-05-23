# Importing base types
from openhsl.base.hsi import HSImage
from openhsl.base.hs_mask import HSMask

#  Importing networks
from openhsl.nn.models.ssftt import SSFTT
from openhsl.nn.models.nm3dcnn import NM3DCNN
from openhsl.nn.models.m1dcnn import M1DCNN
from openhsl.nn.models.m3dcnn_li import M3DCNN as Li
from openhsl.nn.models.tf2dcnn import TF2DCNN

# import support tools
from openhsl.base.hs_mask import draw_colored_mask
from openhsl.nn.models.utils import draw_fit_plots
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from openhsl.nn.data.utils import HyperStandardScaler

# import pca wrapper for hsi
from openhsl.nn.data.utils import apply_pca


#hsi_path = '../demo_data/corn_1.mat'
#hsi_key = 'image'
#mask_path = '../demo_data/mask_corn_1.mat'
#mask_key = 'img'


hsi_path = '../test_data/tr_pr/PaviaU.mat'
hsi_key = 'paviaU'
mask_path = '../test_data/tr_pr/PaviaU_gt.mat'
mask_key = 'paviaU_gt'

hsi = HSImage()
mask = HSMask()

hsi.load(path_to_data=hsi_path, key=hsi_key)
mask.load(path_to_data=mask_path, key=mask_key)

scaler = HyperStandardScaler()

hsi.data = scaler.fit_transform(hsi.data)

#hsi_pca = hsi.data

hsi_pca, _ = apply_pca(hsi.data, 30)

optimizer_params = {
    "learning_rate": 0.1,
    "weight_decay": 0
}

scheduler_params = {
    "step_size": 2,
    "gamma": 0.2
}

augmentation_params = {
    "flip_augmentation": False,
    "radiation_augmentation": False,
    "mixture_augmentation": False
}

fit_params = {
    "epochs": 20,
    "train_sample_percentage": 0.1,
    "dataloader_mode": "fixed",
    "wandb_vis": False,
    "optimizer_params": optimizer_params,
    "batch_size": 128,
    "scheduler_type": 'StepLR',
    "scheduler_params": scheduler_params
}

cnn = M1DCNN(n_classes=mask.n_classes,
             n_bands=hsi_pca.data.shape[-1],  # or hsi.data.shape[-1]
             device='cuda')

# cnn.init_wandb()

cnn.fit(X=hsi_pca,  # or hsi
        y=mask.get_2d(),
        fit_params=fit_params)

draw_fit_plots(model=cnn)

pred = cnn.predict(X=hsi_pca,  # or hsi
                   y=mask,
                   batch_size=100)

pred = pred * (mask.get_2d() > 0)

plt.imshow(pred)
plt.show()

draw_colored_mask(mask=mask,
                  predicted_mask=pred,
                  stack_type='h')

print(classification_report(pred.flatten(), mask.get_2d().flatten()))

