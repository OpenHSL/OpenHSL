# Importing base types
from openhsl.hsi import HSImage
from openhsl.hs_mask import HSMask

#  Importing networks
from openhsl.models.ssftt import SSFTT
from openhsl.models.fcnn import FCNN
# from openhsl.models.hsicnn_luo import HSICNN
from openhsl.models.nm3dcnn import NM3DCNN
from openhsl.models.hsicnn_luo import HSICNN as Luo
from openhsl.models.m1dcnn import M1DCNN
# from openhsl.models.m3dcnn_sharma import M3DCNN as Sharma
# from openhsl.models.m3dcnn_hamida import M3DCNN as Hamida
# from openhsl.models.m3dcnn_he import M3DCNN as He
from openhsl.models.m3dcnn_li import M3DCNN as Li
from openhsl.models.tf2dcnn import TF2DCNN

# import support tools
from openhsl.utils import draw_fit_plots, draw_colored_mask
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

# import pca wrapper for hsi
from openhsl.data.utils import apply_pca


# hsi_path = '../demo_data/corn_1.mat'
# hsi_key = 'image'
# mask_path = '../demo_data/mask_corn_1.mat'
# mask_key = 'img'


hsi_path = '../test_data/tr_pr/PaviaU.mat'
hsi_key = 'paviaU'
mask_path = '../test_data/tr_pr/PaviaU_gt.mat'
mask_key = 'paviaU_gt'

hsi = HSImage()
mask = HSMask()

hsi.load(path_to_data=hsi_path, key=hsi_key)
mask.load(path_to_data=mask_path, key=mask_key)

hsi_pca, _ = apply_pca(hsi.data, 30)

optimizer_params = {
    "learning_rate": 0.001,
    "weight_decay": 0
}

scheduler_params = {
    "step_size": 1,
    "gamma": 0.1
}

augmentation_params = {
    "flip_augmentation": False,
    "radiation_augmentation": False,
    "mixture_augmentation": False
}

fit_params = {
    "epochs": 10,
    "train_sample_percentage": 0.2,
    "dataloader_mode": "fixed",
    "wandb_vis": False,
    # "optimizer_params": optimizer_params,
    "batch_size": 128,
    "scheduler_type": 'StepLR',
    "scheduler_params": scheduler_params
}

cnn = TF2DCNN(n_classes=mask.n_classes,
              n_bands=hsi_pca.data.shape[-1],  # or hsi.data.shape[-1]
              device='cuda')

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
