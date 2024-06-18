# Платформа с открытым кодом для сквозной технологии формирования, обработки и анализа гиперспектральных изображений.

## Сайт проекта

[ссылка](https://openhsl.org/)

## Минимальные технические требования

- ОС: Windows 10 или старше / Ubuntu 20.04 или старше
- GPU: Nvidia RTX 3090 или старше
- не менее 32 ГБ ОЗУ
- Cuda 1.18 или старше
- Python 3.9 или старше

## Формирование ГСИ
--------------------

Для формирования ГСИ из набора кадров, содержащих спектральную развёртку, в форматах AVI, PNG или BMP можно воспользоваться:
1) [CLI версией](https://github.com/OpenHSL/OpenHSL/blob/main/cli_demo_builder.py), задав в этом файле требуемые параметры и затем выполнив команду *python cli_demo_builder.py*
2) GUI версией, выполнив команду *python builder_app.py*


## Разметка ГСИ
--------------------
Для разметки ГСИ можно воспользоваться:
1) [CLI версией](https://github.com/OpenHSL/OpenHSL/blob/main/cli_demo_painter.py), поддерживающей только методы автоматизированной разметки, задав в этом файле требуемые параметры и затем выполнив команду *python cli_demo_painter.py*
2) GUI версией, выполнив команду *python annotator_main.py*

## Нейросетевой анализ ГСИ
---------------------------
Для обучения и инференса нейросетевых классификаторов можно воспользоваться:
1) [CLI версией](https://github.com/OpenHSL/OpenHSL/blob/main/cli_demo_trainer.py), задав в этом файле требуемые параметры и затем выполнив команду *python cli_demo_trainer.py*
2) GUI версией, выполнив команду *python trainer_app.py*


## Примеры работы с наборами данных
-----------------------------------

В качестве тестовых примеров ниже представлены 10 наборов данных, содержащих исходные кадры со спектральной развёрткой, сформированные ГСИ, маски разметки ГСИ и веса моделей. 
Для работы с каждым из представленных наборов пользователю требуется скачать на своё устройство по указанным ссылкам наборы данных и веса моделей.

### 1. Cельскохозяйственные и сорные растения

#### Описание
Данное ГСИ получено щелевым гиперспектрометром, установленным на поворотной платформе. На нём представлены два вида сельскохозяйственных культур (овёс и кукуруза), а также сорное растение, произрастающее между культурными растениями. 
Такой набор данных может применяться в задачах умного сельского хозяйства для обнаружения и классификации сорных растений. 

Характеристики ГСИ:
- пространственное разрешение: 976х3000 пикселей;
- количество спектральных каналов: 250;
- диапазон длин волн: 420-980нм.
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: MAT;
- формат исходных данных: AVI.

Характеристики маски разметки:
- маска разметки содержит в себе 4 класса:
0 – игнорируемый класс;
1 – овёс;
2 – сорняк;
3 – кукуруза;
- формат файла: MAT.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-corn)

Архитектура предобученной модели: TF2DCNN

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/corn_tf2dcnn/blob/main/tf2d_70band_PCA.h5)

#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/corn1.mat', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *corn1.mat* и метаданными *corn1_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/mask_corn_1.mat', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *mask_corn_1.mat* и метаданными *mask_corn_1_metainfo.json*
   - MASK_KEY = 'img'
   - USE_PCA = True
   - NUM_COMPONENTS = 70
   - CLASSIFIER = 'TF2DCNN'
   - PATH_TO_WEIGHTS = '.../tf2d_70band_PCA.h5', где вместо ... указать директорию содержащую файл весов модели *tf2d_70band_PCA.h5*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 2. Три сорта кофе

#### Описание
Данное ГСИ содержит три различных сорта кофе: «Кения – АА Маунт», «Бразилия – Суль де Минас» и «Коста Рика – Тарразу». Получено щелевым гиперспектрометром, установленным на поворотной платформе. 
Такой набор данных может применяться в задачах пищевой безопасности.

Характеристики ГСИ:
- пространственное разрешение: 800х2100 пикселей;
- количество спектральных каналов: 200;
- диапазон длин волн: 530-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: NPY;
- формат исходных данных: AVI.
 
Характеристики маски разметки:
- маска разметки содержит в себе 5 классов:
0 – игнорируемый класс;
1 – сорт «Кения – АА Маунт»;
2 – сорт «Бразилия – Суль де Минас»;
3 – сорт «Коста Рика – Тарразу»;
4 – блики на зёрнах;
- формат файла: NPY.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-three-coffee-varieties)

Архитектура предобученной модели: SSFTT

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/coffee_ssftt/blob/main/ssftt.pth)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/triple_coffee.npy', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *triple_coffee.npy* и метаданными *triple_coffee_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/triple_coffee_mask.npy', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *triple_coffee_mask.npy* и метаданными *triple_coffee_mask_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'SSFTT'
   - PATH_TO_WEIGHTS = '.../ssftt.pth', где вместо ... указать директорию содержащую файл весов модели *ssftt.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 3. Растительный микропрепарат

#### Описание

Данное ГСИ содержит фрагменты микропрепарата поражённого и здорового участков листа растения. Получено [щелевым гиперспектральным микроскопом](https://github.com/OpenHSL/PushBroom_Controller).  
Такой набор данных может применяться в задачах умного сельского хозяйства, экологического мониторинга и пищевой безопасности.

Характеристики ГСИ:
- пространственное разрешение: 1000х1000 пикселей;
- количество спектральных каналов: 200;
- диапазон длин волн: 530-980 нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: TIFF;
- формат исходных данных: PNG.

Характеристики маски разметки:
- маска разметки содержит в себе 3 класса:
0 – игнорируемый класс;
1 – поражённый участок;
2 – здоровый участок;
- формат файла: TIFF.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdata-plant-microscope)

Архитектура предобученной модели: M3DCNN-Li

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/plant_micro_m3dli/blob/main/M3DCNN_Li.pth)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/leaves.tiff', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *leaves.tiff* и метаданными *leaves_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/leaves_mask.tiff', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *leaves_mask.tiff* и метаданными *leaves_mask_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'M3DCNN'
   - PATH_TO_WEIGHTS = '.../M3DCNN_Li.pth', где вместо ... указать директорию содержащую файл весов модели *M3DCNN_Li.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 4. Окрашенный медицинский микропрепарат

#### Описание

Данное ГСИ содержит фрагмент окрашенного микропрепарата артерии млекопитающего с различными клеточными структурами и типами тканей. 
Получено [щелевым гиперспектральным микроскопом](https://github.com/OpenHSL/PushBroom_Controller).  Такой набор данных может применяться в задачах медицины.

Характеристики ГСИ:
- пространственное разрешение: 780х1551 пикселей;
- количество спектральных каналов: 200;
- диапазон длин волн: 530-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: H5;
- формат исходных данных: PNG.
  
Характеристики маски разметки:
- маска разметки содержит в себе 6 классов:
0 – игнорируемый класс;
1 – внутренняя эластичная мембрана и ядра клеток – соединительная ткань;
2 – оболочка гладкомышечных клеток;
3 – эластичные волокна;
4 – кровь;
5 – эндотелиальные клетки;
- формат файла: H5.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-stained-microscope)

Архитектура предобученной модели: NM3DCNN

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/stained_micro_nm3d/blob/main/NM3DCNN.pth)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/artery.h5', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *artery.h5* и метаданными *artery_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/artery_mask.h5', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *artery_mask.h5* и метаданными *leaves_mask_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'NM3DCNN'
   - PATH_TO_WEIGHTS = '.../NM3DCNN.pth', где вместо ... указать директорию содержащую файл весов модели *NM3DCNN.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 5. Неокрашенный медицинский микропрепарат

#### Описание

Данное ГСИ содержит фрагмент неокрашенного микропрепарата сосуда млекопитающего с различными типами тканей. Получено [щелевым гиперспектральным микроскопом](https://github.com/OpenHSL/PushBroom_Controller).  Такой набор данных может применяться в задачах медицины. 

Характеристики ГСИ:
- пространственное разрешение: 1100х980 пикселей;
- количество спектральных каналов: 200;
- диапазон длин волн: 530-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: MAT;
- формат исходных данных: BMP.

Характеристики маски разметки:
- маска разметки содержит в себе 4 класса:
0 – игнорируемый класс;
1 – фрагменты соединительной ткани;
2 – гладкая мышечная ткань;
3 – соединительная ткань;
- формат файла: PNG.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-unstained-tissue-microslide)

Архитектура предобученной модели: NM3DCNN

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/unstained_micro_nm3d/blob/main/NM3DCNN.pth)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/vessel1.mat', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *vessel1.mat* и метаданными *vessel1_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/vessel1_mask.png', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *vessel1_mask.png* и метаданными *vessel1_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'NM3DCNN'
   - PATH_TO_WEIGHTS = '.../NM3DCNN.pth', где вместо ... указать директорию содержащую файл весов модели *NM3DCNN.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 6. Образцы почвы

#### Описание

Данное ГСИ содержит 9 образцов почвы с различным содержанием минеральных веществ (для данного набора использовались значения содержания углерода). 
Получено щелевым гиперспектрометром, установленным над конвейерной лентой.  Такой набор данных может применяться в задачах умного сельского хозяйства и экологического мониторинга.

Характеристики ГСИ:
- пространственное разрешение: 800x1940 пикселей;
- количество спектральных каналов: 250;
- диапазон длин волн: 420-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: MAT;
- формат исходных данных: AVI.

Характеристики маски разметки:
- маска разметки содержит в себе 4 класса:
0 – игнорируемый класс;
1 – низкое содержание углерода;
2 – среднее содержание углерода;
3 – Высокое содержание углерода;
- формат файла: BMP.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-soil)

Архитектура предобученной модели: SSFTT

Веса модели доступны по [ссылке]
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/soil.mat', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *soil.mat* и метаданными *soil_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/carbon_mask.bmp', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *carbon_mask.bmp* и метаданными *carbon_mask_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'SSFTT'
   - PATH_TO_WEIGHTS = '.../SSFTT.pth', где вместо ... указать директорию содержащую файл весов модели *SSFTT.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 7. Сельскохозяйственное и сорное растения

#### Описание

Данное ГСИ содержит два вида растений: клубника и сорное растение. Получено щелевым гиперспектрометром, установленным на поливальном механизме. 
Такой набор данных может применяться в задачах умного сельского хозяйства для классификации целевых видов сельскохозяйственных культур.

Характеристики ГСИ:
- пространственное разрешение: 780х4775 пикселей;
- количество спектральных каналов: 250;
- диапазон длин волн: 420-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: MAT;
- формат исходных данных: AVI.

Характеристики маски разметки:
- маска разметки содержит в себе 3 класса:
0 – игнорируемый класс;
1 – другие растения;
2 – клубника;
- формат файла: MAT.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-strawberry)

Архитектура предобученной модели: M1DCNN

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/strawberry_m1d/blob/main/m1dcnn.pth)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/strawberries.mat', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *strawberries.mat* и метаданными *strawberries_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/strawberries_mask.mat', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *strawberries_mask.mat* и метаданными *strawberries_mask_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'M1DCNN'
   - PATH_TO_WEIGHTS = '.../m1dcnn.pth', где вместо ... указать директорию содержащую файл весов модели *m1dcnn.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 8. Натуральные и пластиковые объекты

#### Описание

Данное ГСИ содержит две сцены с различными видами пластиковых и растительных объектов (например, пластиковый апельсин и настоящий апельсин). 
Получено щелевым гиперспектрометром, установленным на поворотной платформе. Такой набор данных может применяться в задачах умного сельского хозяйства и пищевой безопасности.

Характеристики ГСИ:
- пространственное разрешение: 800х1300 пикселей;
- количество спектральных каналов: 250;
- диапазон длин волн: 420-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: MAT;
- формат исходных данных: AVI.

Характеристики маски разметки:
- маска разметки содержит в себе 8 классов:
0 – игнорируемый класс;
1 – восковая свеча в виде кактуса;
2 – пластиковый апельсин;
3 – пластиковое яблоко;
4 – томат сорт №1;
5 – томат сорт №2;
6 – красная пластиковая крышка;
7 – апельсин;
- формат файла: MAT.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-plastic-and-natural-objects)

Архитектура предобученной модели: M3DCNN-Li

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/natural_and_plastic_m3dli/blob/main/M3DCNN_Li.pth)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/tomato.mat', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *tomato.mat* и метаданными *strawberries_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/tomato_mask.mat', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *tomato_mask.mat* и метаданными *tomato_mask_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'M3DCNN'
   - PATH_TO_WEIGHTS = '.../M3DCNN_Li.pth', где вместо ... указать директорию содержащую файл весов модели *M3DCNN_Li.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 9. Два вида таблеток

#### Описание

Данное ГСИ содержит два вида белый таблеток: белый уголь и мел.  Получено щелевым гиперспектрометром, установленным на поворотной платформе. Такой набор данных может применяться в задачах пищевой безопасности.

Характеристики ГСИ:
- пространственное разрешение: 974x1351 пикселей;
- количество спектральных каналов: 250;
- диапазон длин волн: 420-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: MAT;
- формат исходных данных: AVI.

Характеристики маски разметки:
- маска разметки содержит в себе 3 класса:
0 – игнорируемый класс;
1 – белый уголь;
2 – мел;
- формат файла: MAT.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-white-tablets)

Архитектура предобученной модели: M1DCNN

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/tablets_m1d/blob/main/M1DCNN.pth)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/tablets.mat', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *tablets.mat* и метаданными *tablets_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/tablets_mask.mat', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *tablets_mask.mat* и метаданными *tablets_mask_metainfo.json*
   - MASK_KEY = 'img'
   - CLASSIFIER = 'M1DCNN'
   - PATH_TO_WEIGHTS = '.../M1DCNN.pth', где вместо ... указать директорию содержащую файл весов модели *M1DCNN.pth*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.

### 10. Область водоёма

#### Описание

Данное ГСИ содержит два участка водоема с различным содержанием минеральных веществ.  Оно сформировано из фрагментов двух ГСИ, полученных гиперспектрометром, установленным на БПЛА. Такой набор данных может применяться в задачах экологического мониторинга.

Характеристики ГСИ:
- пространственное разрешение: 720х1420 пикселей;
- количество спектральных каналов: 200;
- диапазон длин волн: 530-980нм;
- спектральные каналы представлены восьмибитными полутоновыми изображениями;
- формат файла: MAT;
- формат исходных данных: AVI.

Характеристики маски разметки:
- маска разметки содержит в себе 4 класса:
0 – игнорируемый класс;
1 – низкое загрязнение;
2 – высокое загрязнение;
- формат файла: MAT.


Набор данных доступен для скачивания по [ссылке](https://www.kaggle.com/datasets/openhsl/hyperdataset-uav-river)

Архитектура предобученной модели: TF2DCNN

Веса модели доступны по [ссылке](https://huggingface.co/OpenHSL/uav_tf2d/blob/main/TF2DCNN_70bands_PCA.h5)
#### Формирование ГСИ из набора кадров
Для формирования данного ГСИ из набора кадров при помощи CLI требуется:
...

#### Инференс на предобученных весах модели
Для демонстрации инференса предобученной модели при помощи CLI требуется:
1) изменить в файле *cli_demo_trainer.py* значения нижепредставленных параметров следующим образом:
   - HSI_PATH = '.../hsi/river.mat', где вместо ... указать директорию содержащую папку *hsi* с файлом ГСИ *river.mat* и метаданными *river_metainfo.json*
   - HSI_KEY = 'image'
   - MASK_PATH = '.../mask/river_mask.mat', где вместо ... указать директорию содержащую папку *mask* с файлом ГСИ *river_mask.mat* и метаданными *river_mask_metainfo.json*
   - MASK_KEY = 'img'
   - USE_PCA = True
   - NUM_COMPONENTS = 70
   - CLASSIFIER = 'TF2DCNN'
   - PATH_TO_WEIGHTS = '.../TF2DCNN_70bands_PCA.h5', где вместо ... указать директорию содержащую файл весов модели *TF2DCNN_70bands_PCA.h5*
   - PREDICT_ONLY = True
2) Остальные параметры оставить по-умолчанию;
3) выполнить команду *python cli_demo_trainer.py*.
