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

# ----------------------------------------------------------------------------------------------------------------------

# Указывается путь до файла ГСИ, допустимые форматы MAT, NPY, TIFF, H5
HSI_PATH = '...'

# Указывается ключ файла ГСИ, если формат MAT или H5, иначе None
HSI_KEY = '...'

# Указывается путь до файла маски разметки ГСИ, допустимые форматы MAT, NPY, TIFF, H5, PNG, BMP
MASK_PATH = '...'

# Указывается ключ файла маски разметки ГСИ, если формат MAT или H5, иначе None
MASK_KEY = '...'

# ----------------------------------------------------------------------------------------------------------------------

# Указывается тип скалера 'Standard' или 'MinMax'
SCALER_TYPE = 'Standard'

# Указывается ось для скалера 'band' или 'pixel'
SCALE_PER = 'band'

# ----------------------------------------------------------------------------------------------------------------------
# Указывается False или True для применения PCA
USE_PCA = False

# Указывается количество компонент PCA
NUM_COMPONENTS = 70
# ----------------------------------------------------------------------------------------------------------------------

# Указывается тип нейросетевого классификатора, доступны для выбора: 'M1DCNN', 'TF2DCNN', 'M3DCNN', 'NM3DCNN', 'SSFTT'
CLASSIFIER = 'M1DCNN'

# Указывается путь до весов предобученной модели или None
PATH_TO_WEIGHTS = None

# ----------------------------------------------------------------------------------------------------------------------

# Указывается False или True для использования в качестве RT API WandB.
# Для использования требуется заполненный файл wandb.yaml
USE_WANDB = False

# ----------------------------------------------------------------------------------------------------------------------

# Указывается False или True для случая, когда не требуется обучение
PREDICT_ONLY = True

# ----------------------------------------------------------------------------------------------------------------------

# Указывается шаг обучения
OPTIM_LEARNING_RATE = 0.01

# Указывается коэффициент сокращения весов
OPTIM_WEIGHT_DECAY = 0

# ----------------------------------------------------------------------------------------------------------------------

# Указывается тип шедулера, доступные значения 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
SCHEDULER_TYPE = 'StepLR'

# Указывается количество эпох, после которого шедулер уменьшает шаг обучения
SCHED_STEP_SIZE = 5

# Указывается коэффициент уменьшения шага обучения
SCHED_GAMMA = 0.5

# ----------------------------------------------------------------------------------------------------------------------

# Указывается количество эпох обучения
EPOCHS = 50

# Указывается доля данных из ГСИ для обучения
TRAIN_SAMPLE_PERCENTAGE = 0.1

# Указывается режим работы даталоадера, доступные 'random', 'fixed', 'disjoint'
DATALOADER_MODE = '...'

# Указывается размер батча обучения
BATCH_SIZE = 32
# ----------------------------------------------------------------------------------------------------------------------

# Указывается
f1_type = '...'

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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

net = _CLASSIFIER(n_classes=mask.n_classes,
                  n_bands=hsi.data.shape[-1],  # or hsi.data.shape[-1]
                  path_to_weights=PATH_TO_WEIGHTS,
                  device='cuda')

if USE_WANDB:
    net.init_wandb()

if not PREDICT_ONLY:
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
