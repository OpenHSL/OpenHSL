import json
from types import SimpleNamespace
import os
import time
import cv2 as cv
import numpy as np

from src.Utils import Utils
from src.model.Layer import Layer

global MASK_IND

class ProjectModel:
    def __init__(self):
        self._config_file_path = "MaskPainter_config.json"
        self._comp_dir = "comp"
        self._mask_dir = "mask"

        self.default_background_color = "#3399ff"

        self.projectPath = None          # ie. 'C:/some_path/app_root/projects/example_project.json'
        self.projectFileName = None      # ie. 'example_project.json'
        self.projectName = None          # ie. 'example_project'
        self.compRootDir = None          # ie. 'C:/some_path/app_root/projects/comp/'
        self.maskRootDir = None          # ie. 'C:/some_path/app_root/projects/mask/'

        self.backgroundImagePath = None
        self.cvBackgroundImage = None

        self.activeLayer = None
        self.activeMask = -1
        self.maskOpaque = False

        self.layers = dict()
        self.layerKeys = []

        self.imgSize = None
        self.numMasks = None
        
        self.hsi_MAXhsilayer = 0

    def unload(self):
        self.__init__()

    @staticmethod
    def _generate_layer_uid():
        return round(time.time() * 1000)

    @staticmethod
    def _generate_comp_file_name(image_prefix):
        return "{}_comp.png".format(image_prefix)

    @staticmethod
    def _generate_mask_file_name(image_prefix, image_key):
        return "{}_mask{}.jpg".format(image_prefix, image_key)

    @staticmethod
    def _generate_background_file_name(image_prefix):
        return "{}_background.jpg".format(image_prefix)

    @staticmethod
    def _generate_large_background_file_name(image_prefix):
        return "{}_large_background.jpg".format(image_prefix)

    @staticmethod
    def _write_default_config(config_file_path):
        dictionary = {
            "default_project_settings": {
                "mask_width": 640,
                "mask_height": 360,
                "max_background_width": 1600,
                "max_background_height": 900,
                "layer_keys": [],
                "layers": {},
                "active_mask": -1,
                "mask_opaque": True
            }
        }
        with open(config_file_path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)
        outfile.close()

    @staticmethod
    def _read_json_file(file_path):
        f = open(file_path)
        obj = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        f.close()
        return obj

    def get_layer_by_z(self, z):
        return self.get_layer_by_uid(self.layerKeys[z])

    def get_layer_by_uid(self, uid):
        return self.layers[uid]

    def get_visibility_list(self):
        vis = []
        for z in range(self.numMasks):
            vis.append(self.get_layer_by_z(z).isVisible)
        return vis

    def export_comp_image(self, pil_image):
        if not os.path.exists(self.compRootDir):
            os.mkdir(self.compRootDir)
        outfile = os.path.join(self.compRootDir, self._generate_comp_file_name(self.projectName))
        pil_image.save(outfile)

    def save_as(self, project_file_path):
        self._set_paths(project_file_path)
        self._save_project_json()
        self._save_masks()

        # model will use subject to update observer views

    def save(self):
        self._save_project_json()
        self._save_masks()

    def create_project(self, project_file_path, bg_image_file_path=None, key_answer="paviaU"):
        self.unload()
        self._set_paths(project_file_path)

        config = self._read_default_config(self._config_file_path)
        bg_img_size = self._copy_background_image(config, bg_image_file_path, key_answer)
        self._process_config(config, bg_img_size)

        self._save_project_json()
        self._save_masks()

        # model will use subject to update observer views

    def load_project(self, project_file_path):
        self.unload()
        self._set_paths(project_file_path)
        self._read_project_json(project_file_path)

        # model will use subject to update observer views

    def _set_paths(self, project_file_path):
        self.projectPath = project_file_path
        project_root_dir, self.projectFileName = os.path.split(self.projectPath)
        self.projectName = self.projectFileName.split('.')[0]
        self.compRootDir = os.path.join(project_root_dir, self._comp_dir)
        self.maskRootDir = os.path.join(project_root_dir, self._mask_dir)

    def _read_default_config(self, config_file_path):
        if not os.path.exists(config_file_path):
            self._write_default_config(config_file_path)

        obj = self._read_json_file(config_file_path)
        return obj.default_project_settings

    def _copy_background_image(self, config, bg_image_file_path, key_answer):
        # returns img_size (w, h) if bg_image exists, otherwise returns None
        if bg_image_file_path:
            if not os.path.exists(self.maskRootDir):
                os.mkdir(self.maskRootDir)
                
            #\
            
            from matplotlib import pyplot as plt
            import matplotlib.cm as cm
            from PIL import Image as im
            from hsi import HSImage # ВСКРЫТЬ / ПОФИКСИТЬ
            
            hsi_data = HSImage()
            hsi_data.load_from_mat(path_to_file=bg_image_file_path, mat_key= key_answer) # "image" прикрутить ползунок запрашивать у пользователя мат-кей paviaU  / # добавить окно спросить у пользователя, какой ключ выбрать - 
            
            #plt.imsave('filename.png', np.array(hsi_data[10]).reshape(610,340), cmap=cm.magma) # viridis
            
            #/ np.array(hsi_data).shape (250, 720, 1900)
            self.hsi_MAXhsilayer =  np.array(hsi_data).shape[0]
            hsi_h =  np.array(hsi_data).shape[1]
            hsi_w =  np.array(hsi_data).shape[2]
            
            data = np.array(hsi_data[100]) # data = np.array(hsi_data[5], np.uint8).reshape(610,340)       int32
           
            
            #img_numpy = np.array(hsi_data[5], dtype=np.uint8)
            #img = im.fromarray(img_numpy, "RGB")
            
            #cv_bg = cv.imread(bg_image_file_path)
            print(data.shape) # data.shape (720, 1900)
            #cv_bg = im.fromarray(data) # , 'RGB'

            cv_bg = data
            
            h = cv_bg.shape[0]
            w = cv_bg.shape[1]

            # downscale the image if it is more than config's max bg size
            h_over = h / config.max_background_height
            w_over = w / config.max_background_width
            over = max(h_over, w_over)
            if over > 1.0:
                large_bg_filename = self._generate_large_background_file_name(self.projectName)
                large_bg_path = os.path.join(self.maskRootDir, large_bg_filename)
                cv.imwrite(large_bg_path, cv_bg)

                h = int(h / over)
                w = int(w / over)
                cv_bg = cv.resize(cv_bg, (w, h))

            bg_filename = self._generate_background_file_name(self.projectName)
            self.backgroundImagePath = os.path.join(self.maskRootDir, bg_filename)
            cv.imwrite(self.backgroundImagePath, cv_bg)

            return w, h
        return None

    def _process_config(self, config, bg_img_size):
        if bg_img_size:
            w, h = bg_img_size
        else:
            h = config.mask_height
            w = config.mask_width
        self.imgSize = (w, h)

        # derive data from config
        self.maskOpaque = config.mask_opaque
        self.activeMask = config.active_mask
        self.numMasks = len(config.layer_keys)
        self.layerKeys = config.layer_keys
        layers = config.layers.__dict__

        for i in range(self.numMasks):
            k = self.layerKeys[i]
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            layer = layers[k].__dict__
            self.layers[k] = Layer(layer, mask)

        if self.numMasks == 0:
            self.insert_layer(0, "#ff00ff")
            self.set_active_layer(0)

    def _read_project_json(self, project_file_path):
        obj = self._read_json_file(project_file_path)

        # use a background image if it exists in the mask dir
        if os.path.exists(self.maskRootDir):
            bg_filename = self._generate_background_file_name(self.projectName)
            bg_file_path = os.path.join(self.maskRootDir, bg_filename)
            if os.path.exists(bg_file_path):
                cv_bg = cv.imread(bg_file_path)
                self.cvBackgroundImage = cv.cvtColor(cv_bg, cv.COLOR_BGR2RGBA)
                self.backgroundImagePath = bg_file_path

        # derive data from project file
        self.activeMask = obj.active_mask
        self.maskOpaque = obj.mask_opaque
        self.numMasks = len(obj.layer_keys)
        self.layerKeys = obj.layer_keys
        layers = obj.layers.__dict__

        for i in range(self.numMasks):
            k = self.layerKeys[i]
            mask_filename = self._generate_mask_file_name(self.projectName, k)
            mask_path = os.path.join(self.maskRootDir, mask_filename)
            cv_mask_bgr = cv.imread(mask_path)
            cv_mask = cv.cvtColor(cv_mask_bgr, cv.COLOR_BGR2GRAY)

            layer = layers[k].__dict__
            self.layers[k] = Layer(layer, cv_mask)

            if self.activeMask == i:
                self.activeLayer = self.layers[k]

        # fill in the missing data
        mask0 = self.layers[self.layerKeys[0]].cvMask
        self.imgSize = (mask0.shape[1], mask0.shape[0])

    def _save_project_json(self):
        # build a json from python data
        # save json to file

        json_layers = {}
        for i in range(self.numMasks):
            k = self.layerKeys[i]
            layer = self.layers[k]
            json_layers[k] = {"name": layer.name,
                              "color": layer.color,
                              "visible": layer.isVisible,
                              "locked": layer.isLocked}

        dictionary = {
            "project_name": self.projectName,
            "layer_keys": self.layerKeys,
            "layers": json_layers,
            "active_mask": self.activeMask,
            "mask_opaque": self.maskOpaque
        }

        with open(self.projectPath, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)
        outfile.close()

    def _save_masks(self):
        if not os.path.exists(self.maskRootDir):
            os.mkdir(self.maskRootDir)
        for i in range(self.numMasks):
            k = self.layerKeys[i]
            mask_filename = self._generate_mask_file_name(self.projectName, k)
            cv.imwrite(os.path.join(self.maskRootDir, mask_filename), self.layers[k].cvMask)

    def insert_layer(self, z, color=None):
        if not color:
            if z == 0:
                color = self.get_layer_by_z(z).color
            else:
                if z == self.numMasks:
                    color = self.get_layer_by_z(z-1).color
                else:
                    color = Utils.average_hex_colors(self.get_layer_by_z(z-1).color, self.get_layer_by_z(z).color)

        w, h = self.imgSize
        cv_mask_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        cv_mask = cv.cvtColor(cv_mask_bgr, cv.COLOR_BGR2GRAY)
        uid = self._generate_layer_uid()

        json_layer = {"name": "New Layer",
                      "color": color,
                      "visible": True,
                      "locked": False}
        self.layers[str(uid)] = Layer(json_layer, cv_mask)
        if z < self.numMasks:
            self.layerKeys.insert(z, str(uid))
        else:
            self.layerKeys.append(str(uid))
        self.numMasks = len(self.layerKeys)

        if self.activeMask >= z:
            self.set_active_layer(self.activeMask + 1)

    def remove_layer(self, z):
        uid = self.layerKeys[z]
        self.layerKeys.pop(z)
        del self.layers[uid]

        self.numMasks = len(self.layerKeys)
        if self.activeMask == z:
            self.set_active_layer(0)
        elif self.activeMask > z:
            self.set_active_layer(self.activeMask - 1)

    # refactored here from LayerModel
    def set_active_layer(self, z):
        self.activeMask = z
        self.activeLayer = self.get_layer_by_z(z)
        self.activeLayer.isVisible = True

    # def set_layer_visibility(self, z, vis):
    #     self.get_layer_by_z(z).isVisible = vis

    def toggle_layer_visibility(self, z):
        layer = self.get_layer_by_z(z)
        layer.isVisible = not layer.isVisible

    def toggle_layer_lock(self, z):
        layer = self.get_layer_by_z(z)
        layer.isLocked = not layer.isLocked

    def toggle_mask_opacity(self):
        self.maskOpaque = not self.maskOpaque

    def move_active(self, before, after):
        if before != after:
            uid = self.layerKeys.pop(before)
            if after < self.numMasks - 1:
                self.layerKeys.insert(after, str(uid))
            else:
                self.layerKeys.append(str(uid))
            self.activeMask = after


