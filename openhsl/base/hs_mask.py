import h5py
import json
import matplotlib.patches as mpatches
import numpy as np
import os.path
import rasterio
import seaborn as sns

from matplotlib import pyplot as plt
from PIL import Image
from scipy.io import loadmat, savemat
from typing import Dict, Literal, Optional, Union


class HSMask:
    """
    HSMask()
        Image-like object which contains:
            2D-Array
                Each pixel has value from [0, class_counts - 1]
            3D-Array
                Each layer is binary image where 1 is class and 0 is not-class

        Parameters
        ----------
        mask: np.ndarray
            3D-matrix which has a dimension X - Y - Z.
            where:
                X, Y data resolution.
                Z is a count of channels (1, 3, 4).
        label_class: dict
            dictionary where keys are number of the binary layer in mask
            and values are description class of this layer

        Attributes
        ----------
        data: np.ndarray

        label_class: dict

        Examples
        --------
            arr = np.zeros((100, 100, 3))
            md = {'1':'class_1', '2':'class_2'}

            hsi = HSMask(hsi=arr, metadata=md)
    """

    def __init__(self,
                 mask: Optional[np.array] = None,
                 label_class: Optional[Dict] = None):
        if np.any(mask):
            if HSMask.__is_correct_2d_mask(mask):
                print("got 2d mask")
                self.data = HSMask.convert_2d_to_3d_mask(mask)

            elif HSMask.__is_correct_3d_mask(mask):
                print("got 3d mask")
                self.data = mask
            else:
                print("Void data or incorrect data. Set data and label classes to None and None")
                self.data = None

            if np.any(self.data) and HSMask.__is_correct_class_dict(d=label_class, class_count=self.data.shape[-1]):
                self.label_class = label_class
            else:
                print("Void data or incorrect data. Set label classes to None")
                self.label_class = None

            self.n_classes = self.data.shape[-1]
        else:
            print("Created void mask")
            print("Class labeles is empty")
            self.data = None
            self.label_class = None
    # ------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, item):
        if item < len(self):
            return self.data[:, :, item]
        else:
            raise IndexError(f"{item} is too much for {len(self)} channels in hsi")
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        if self.data is not None:
            return self.data.shape[-1]
        else:
            return 0
    # ------------------------------------------------------------------------------------------------------------------

    def get_2d(self) -> np.ndarray:
        """
        get_2d()
            returns 2d-mask with values in [0,1,2...]

        """
        return HSMask.convert_3d_to_2d_mask(self.data)
    # ------------------------------------------------------------------------------------------------------------------

    def get_3d(self) -> np.ndarray:
        """
        get_3d()
            returns 3d-mask where each layer (Z-axe) is binary image
        """
        return self.data
    # ------------------------------------------------------------------------------------------------------------------

    def __update_label_class(self, label_class: Dict):
        if HSMask.__is_correct_class_dict(d=label_class,
                                          class_count=len(self.data)):
            self.label_class = label_class
    # ------------------------------------------------------------------------------------------------------------------

    def delete_layer(self, pos: int):
        """
        delete_layer(pos)
            deletes layer in mask by index
            Parameters
            ----------
            pos: int
                layer number for deleting
        """
        tmp_list = list(np.transpose(self.data, (2, 0, 1)))
        tmp_list.pop(pos)
        self.data = np.transpose(np.array(tmp_list), (1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def add_void_layer(self,
                       pos: int = 0,
                       shape=None):
        """
        add_void_layer(pos)
            adds filled by zeros layer in mask by index
            Parameters
            ----------
            pos: int
                layer position for adding
            shape: tuple
                shape of void layer
        """
        if np.any(self.data):
            tmp_list = list(np.transpose(self.data, (2, 0, 1)))
            tmp_list.insert(pos, np.zeros(self.data.shape[:-1], dtype="uint8"))
            self.data = np.transpose(np.array(tmp_list), (1, 2, 0))
        else:
            if not shape:
                raise ValueError('Void shape')
            self.data = np.transpose(np.array([np.zeros(shape)]), (1, 2, 0))
    # ------------------------------------------------------------------------------------------------------------------

    def add_completed_layer(self, pos: int, layer: np.ndarray):
        """
        add_completed_layer(pos, layer)
            adds filled by completed layer in mask by index
            Parameters
            ----------
            pos: int
                layer position for adding
            layer: np.ndarray
                binary layer
        """
        if self.__is_correct_binary_layer(layer):
            tmp_list = list(np.transpose(self.data, (2, 0, 1)))
            tmp_list.insert(pos, layer)
            self.data = np.transpose(np.array(tmp_list), (1, 2, 0))
        else:
            raise ValueError("Incorrect layer!")
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __is_correct_2d_mask(mask: np.ndarray) -> bool:
        """
        __is_correct_2d_mask(mask)
            2D mask must have class values as 0,1,2...
            minimal is 0 and 1 (binary image)

            Parameters
            ----------
            mask: np.ndarray

        """
        # input mask must have 2 dimensions
        valid_types = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
        return len(mask.shape) == 2 and mask.dtype in valid_types
    # ------------------------------------------------------------------------------------------------------------------

    def __is_correct_binary_layer(self, layer: np.ndarray) -> bool:
        """
        __is_correct_binary_layer(layer)
            checks is input layer has only binary values (0 and 1) or not

            Parameters
            ----------
            layer: np.ndarray
        """
        return np.all(layer.shape == self.data.shape[:-1]) and np.all(np.unique(layer) == np.array([0, 1]))
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __is_correct_3d_mask(mask: np.ndarray) -> bool:
        """
        __is_correct_3d_mask(mask)
            3D mask must have class values as binary image in N-layers
            Each layer must be binary!
            minimal is two-layer image

            Parameters
            ----------
            mask: np.ndarray

            Returns
            -------
        """
        # check 3D-condition and layer (class) count
        if len(mask.shape) != 3 and mask.shape[-1] < 2:
            return False

        # check each layer that it's binary
        for layer in np.transpose(mask, (2, 0, 1)):
            if np.all(np.unique(layer) != np.array([0, 1])):
                return False

        return True
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __is_correct_class_dict(d: dict, class_count: int) -> bool:
        """
        __is_correct_class_dict(d, class_count)
            checks class descriptions in input dictionary
            Parameters
            ----------
            d: dict
            class_count: int
        """

        if not d:
            return False

        if len(d) != class_count and np.all(np.array(d.keys()) != np.array(range(0, class_count))):
            return False

        return True
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def convert_2d_to_3d_mask(mask: np.ndarray) -> np.ndarray:
        """
        convert_2d_to_3d_mask(mask)
            returns 3d mask consists binary layers from 2d mask

            Parameters
            ----------
            mask: np.ndarray
        """
        h, w = mask.shape
        count_classes = np.max(mask) + 1
        mask_3d = np.zeros((h, w, count_classes))

        for cl in np.unique(mask):
            mask_3d[:, :, cl] = (mask == cl).astype('uint8')

        return mask_3d
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def convert_3d_to_2d_mask(mask: np.ndarray) -> np.ndarray:
        mask_2d = np.zeros(mask.shape[:2])
        for cl, layer in enumerate(np.transpose(mask, (2, 0, 1))):
            mask_2d[layer == 1] = cl

        return mask_2d.astype('uint8')
    # ------------------------------------------------------------------------------------------------------------------

    def load(self,
             path_to_data: str,
             key: str = None):

        """
        load_mask(path_to_file, mat_key, h5_key)

            Reads information from a file,
            converting it to the numpy.ndarray format

            input data shape:
            ____________
            3-dimensional images in png, bmp, jpg
            format or h5, math, npy files are submitted to the input
            ____________

            Parameters
            ----------
            path_to_data: str
                Path to file
            key: str
                Key for field in .mat and .h5 file as dict object
                file['image']
        """

        _, file_extension = os.path.splitext(path_to_data)

        if file_extension in ['.jpg', '.jpeg', '.bmp', '.png']:
            self.load_from_image(path_to_data=path_to_data)

        elif file_extension == '.npy':
            self.load_from_npy(path_to_data=path_to_data)

        elif file_extension == '.mat':
            self.load_from_mat(path_to_data=path_to_data,
                               key=key)

        elif file_extension == '.h5':
            self.load_from_h5(path_to_data=path_to_data,
                              key=key)

        elif file_extension == '.tiff' or file_extension == '.tif':
            self.load_from_tiff(path_to_data=path_to_data)
        else:
            raise ValueError("unsupported extension")
        # updates number of classes after loading mask
        self.n_classes = self.data.shape[-1]
        self.load_class_info(path_to_data)
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_mat(self, path_to_data, key):
        tmp_data = loadmat(path_to_data)[key]
        if HSMask.__is_correct_2d_mask(tmp_data):
            self.data = HSMask.convert_2d_to_3d_mask(tmp_data)
        elif HSMask.__is_correct_3d_mask(tmp_data):
            self.data = tmp_data
        else:
            raise ValueError("Unsupported type of mask")
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_image(self, path_to_data):
        img = Image.open(path_to_data).convert("L")
        img = np.array(img)
        if HSMask.__is_correct_2d_mask(img):
            self.data = HSMask.convert_2d_to_3d_mask(img)
        else:
            raise ValueError("Not supported image type")
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_h5(self,
                     path_to_data,
                     key='img'):
        tmp_data = h5py.File(path_to_data, 'r')[key]
        if HSMask.__is_correct_2d_mask(tmp_data):
            self.data = HSMask.convert_2d_to_3d_mask(tmp_data)
        elif HSMask.__is_correct_3d_mask(tmp_data):
            self.data = tmp_data
        else:
            raise ValueError("Unsupported type of mask")
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_tiff(self,
                       path_to_data):
        with rasterio.open(path_to_data) as raster:
            tmp_data = raster.read()
            tmp_data = tmp_data.transpose((1, 2, 0))
        if HSMask.__is_correct_2d_mask(tmp_data):
            self.data = HSMask.convert_2d_to_3d_mask(tmp_data)
        elif HSMask.__is_correct_3d_mask(tmp_data):
            self.data = tmp_data
        else:
            raise ValueError("Unsupported type of mask")
    # ------------------------------------------------------------------------------------------------------------------

    def load_from_npy(self,
                      path_to_data):
        tmp_data = np.load(path_to_data)
        if HSMask.__is_correct_2d_mask(tmp_data):
            self.data = HSMask.convert_2d_to_3d_mask(tmp_data)
        elif HSMask.__is_correct_3d_mask(tmp_data):
            self.data = tmp_data
        else:
            raise ValueError("Unsupported type of mask")
    # ------------------------------------------------------------------------------------------------------------------

    def save(self,
             path_to_file: str,
             key: Optional[str] = 'img'):

        pth = os.path.dirname(path_to_file)
        if not os.path.exists(pth):
            os.mkdir(pth)

        if path_to_file.endswith('.mat'):
            self.save_to_mat(path_to_file=path_to_file, mat_key=key)
        elif path_to_file.endswith('.h5'):
            self.save_to_h5(path_to_file=path_to_file, h5_key=key)
        elif path_to_file.endswith('.tiff'):
            self.save_to_tiff(path_to_file=path_to_file)
        elif path_to_file.endswith('.npy'):
            self.save_to_npy(path_to_file=path_to_file)
        elif path_to_file.endswith('.png') or path_to_file.endswith('.bmp'):
            self.save_to_images(path_to_save_file=path_to_file)
        else:
            raise Exception('Unsupported extension')
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_mat(self,
                    path_to_file: str,
                    mat_key: str):
        """
        save_to_mat(path_to_file, mat_key)

            ____________
            save the mask in mat format
            ____________

            Parameters
            ----------
            path_to_file: str
                Path to file
            mat_key: str
                Key for field in .mat file as dict object
                mat_file['image']
        """
        temp_dict = {mat_key: self.data}
        savemat(path_to_file, temp_dict)
        self.save_class_info(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_h5(self,
                   path_to_file: str,
                   h5_key: str):
        """
        save_to_h5(path_to_file, h5_key)

        ____________
        save the mask in h5 format
        ____________

        Parameters
        ----------
        path_to_file: str
            Path to file
        h5_key: str
            Key for field in .mat file as dict object
            mat_file['image']
        h5_key: str
            Key for field in .h5 file as 5h object
        """

        with h5py.File(path_to_file, 'w') as f:
            f.create_dataset(h5_key, data=self.data)
        self.save_class_info(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------
    
    def save_to_npy(self,
                    path_to_file: str):
        """
        save_to_npy(path_to_file)

        ____________
        save the mask in numpy format
        ____________

        Parameters
        ----------
        path_to_file: str
            Path to file
        """
        np.save(path_to_file, self.data)
        self.save_class_info(path_to_file)
    # ------------------------------------------------------------------------------------------------------------------

    def save_to_tiff(self,
                     path_to_file: str):
        if not path_to_file.endswith('.tif') and not path_to_file.endswith('.tiff'):
            raise Exception('Incorrect file format')

        dt = 'int8'
        if self.data.dtype.name == 'uint8' or self.data.dtype.name == 'int8':
            dt = 'int8'
        elif self.data.dtype.name == 'uint16' or self.data.dtype.name == 'int16':
            dt = 'int16'
        elif self.data.dtype.name == 'uint32' or self.data.dtype.name == 'int32':
            dt = 'int32'

        d = {'driver': 'GTiff',
             'dtype': dt,
             'nodata': None,
             'width': self.data.shape[1],
             'height': self.data.shape[0],
             'count': self.data.shape[2],
             'interleave': 'band'}

        with rasterio.open(path_to_file, 'w', **d) as dst:
            dst.write(self.data.transpose((2, 0, 1)))
        self.save_class_info(path_to_file)

    def save_to_images(self,
                       path_to_save_file: str):
        """
        save_image(path_to_save_file)

        ____________
        save the mask in 'jpg','jpeg','bmp','png' format
        ____________

        Parameters
        ----------
        path_to_save_file: str
            Path to file
        """
        img_2d = self.get_2d()
        img = Image.fromarray(img_2d)
        img.save(path_to_save_file)
        self.save_class_info(path_to_save_file)
    # ------------------------------------------------------------------------------------------------------------------

    def load_class_info(self, path_to_data):
        path_to_data = '.'.join(path_to_data.split('.')[:-1]) + '_metainfo.json'
        if os.path.exists(path_to_data):
            with open(path_to_data, 'r') as json_file:
                data = json.load(json_file)
            self.label_class = data['label_class']
        else:
            print("Metainfo file does not exist!")
            self.label_class = {}
    # ------------------------------------------------------------------------------------------------------------------

    def save_class_info(self, path_to_data):
        path_to_data = '.'.join(path_to_data.split('.')[:-1]) + '_metainfo.json'
        if not self.label_class:
            print('Wavelengths are empty! Save as empy dict')
            self.label_class = {}
        data = {"label_class": self.label_class}
        with open(path_to_data, 'w') as outfile:
            outfile.write(json.dumps(data))
    # ------------------------------------------------------------------------------------------------------------------


def get_palette(num_classes):
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_classes)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

    return palette
# ----------------------------------------------------------------------------------------------------------------------


def convert_to_color(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        palette = get_palette(np.max(arr_2d))

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
# ----------------------------------------------------------------------------------------------------------------------


def colorize_mask(mask: Union[HSMask, np.ndarray], palette=None):
    # TODO rethink it
    if isinstance(mask, HSMask):
        mask_data = mask.get_2d()
    elif isinstance(mask, np.ndarray):
        mask_data = mask
    else:
        raise Exception("Unsupported mask type")

    if palette is None:
        palette = get_palette(np.max(mask_data))

    colored_mask = convert_to_color(mask_data, palette=palette)

    return colored_mask
# ----------------------------------------------------------------------------------------------------------------------


def draw_colored_mask(mask: HSMask,
                      predicted_mask: np.array = None,
                      mask_labels: dict = None,
                      stack_type: Literal['v', 'h'] = 'v'):
    # TODO rethink it
    palette = get_palette(np.max(mask.get_2d()))

    color_gt = convert_to_color(mask.get_2d(), palette=palette)

    t = 1
    tmp = lambda x: [i / 255 for i in x]
    cmap = {k: tmp(rgb) + [t] for k, rgb in palette.items()}

    if mask_labels:
        labels = mask_labels
    else:
        labels = mask.label_class

    patches = [mpatches.Patch(color=cmap[i], label=labels.get(str(i), 'no information')) for i in cmap]

    plt.figure(figsize=(12, 12))
    if np.any(predicted_mask):
        color_pred = convert_to_color(predicted_mask, palette=palette)
        if stack_type == 'v':
            combined = np.vstack((color_gt, color_pred))
        elif stack_type == 'h':
            combined = np.hstack((color_gt, color_pred))
        else:
            raise Exception(f'{stack_type} is unresolved mode')
        plt.imshow(combined, label='Colored ground truth and predicted masks')
    else:
        plt.imshow(color_gt, label='Colored ground truth mask')
    if labels:
        plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.show()

    return color_gt
# ----------------------------------------------------------------------------------------------------------------------
