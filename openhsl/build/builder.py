import numpy as np
from openhsl.hsi import HSImage
from openhsl.build.raw_pb_data import RawData
from openhsl.build.uav_builder import build_hypercube_by_videos
from typing import Optional, Dict
from openhsl.utils import gaussian

import cv2
import json
import math
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


class HSBuilder:
    """
    HSBuilder(path_to_data, path_to_metadata=None, data_type=None)

        Build a HSI object from HSRawData

        Parameters
        ----------
        path_to_data : str
            path to hyperspectral source data, for example path to directory with images or videos,
            or path to video-file
        path_to_metadata : str, default=None
            path to file with telemetry data (for example gps-file)
        data_type : str
            'images'
            'video'

        Attributes
        ----------
        path_to_data : str
        path_to_metadata : str
        hsi : HSImage, default=None
        frame_iterator: RawData

        Methods
        -------
        # TODO Must be described

        Examples
        --------
        # TODO Must be described

    """

    def __init__(self,
                 path_to_data: str,
                 path_to_gps: str = None,
                 path_to_metadata: str = None,
                 data_type: str = None):
        if not isinstance(path_to_data, str):
            raise TypeError(f"path_to_data must be str, not {type(path_to_data)}")
        if not isinstance(data_type, str):
            raise TypeError(f"data_type must be str, not {type(data_type)}")
        if path_to_gps and not isinstance(path_to_gps, str):
            raise TypeError(f"path_to_metadata must be str, not {type(path_to_gps)}")
        if data_type not in ['images', 'video']:
            raise ValueError(f"data_type must be 'images' or 'video', not {data_type}")
        self.path_to_data = path_to_data
        self.path_to_gps = path_to_gps
        self.path_to_metadata = path_to_metadata

        if self.path_to_metadata:
            self.__get_metainfo()

        self.hsi: Optional[HSImage] = None
        self.frame_iterator = RawData(path_to_data=path_to_data, type_data=data_type)
    # ------------------------------------------------------------------------------------------------------------------

    def __get_metainfo(self):
        with open(self.path_to_metadata) as f:
            d = json.load(f)
        self.roi_coords = d.get('roi', None)
        self.light_coeff = np.array(d.get('light_norm', None))
    # ------------------------------------------------------------------------------------------------------------------

    # TODO move?
    @staticmethod
    def __norm_frame_camera_illumination(frame: np.ndarray,
                                         light_coeff: np.ndarray) -> np.ndarray:
        """
        __norm_frame_camer_illumination(frame, light_coeff)

            Normalizes illumination on frame.
            Frame have heterogeneous illumination due slit defects. It must be corrected.
            frame shape and light_coeff shape must be equal!

            Parameters
            ----------
            frame : np.ndarray
                hyperspectral frame
            light_coeff : np.ndarray
                light distribution on matrix
            Returns
            -------
                np.ndarray
        """
        if frame.shape != light_coeff.shape:
            raise Exception("frame shape and light_coeff shape must be equal!\n",
                            f"frame shape is: {frame.shape}\n",
                            f'light shape is: {light_coeff.shape}\n')

        return np.multiply(frame.astype("uint8"), light_coeff).astype("uint8")
        #return np.multiply(frame, light_coeff) * np.tile(frame[frame.shape[0] // 2, :], (np.shape(frame)[0], 1))
    # ------------------------------------------------------------------------------------------------------------------

    # TODO move?
    @staticmethod
    def __get_slit_angle(frame: np.ndarray) -> float:
        """
        __get_slit_angle(frame)

            Returns slit tilt angle in degrees (nor radians!)

            Parameters
            ----------
            frame : np.ndarray
                hyperspectral frame

            Returns
            -------
                float
        """
        bl = cv2.GaussianBlur(frame, (11, 11), cv2.BORDER_DEFAULT)
        frame_t = bl > 0.999 * np.max(bl)
        y, x = np.where(frame_t > 0)
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        ang = math.degrees(math.atan(lr.coef_))
        return ang
    # ------------------------------------------------------------------------------------------------------------------

    # TODO move?
    @staticmethod
    def __norm_rotation_frame(frame: np.ndarray) -> np.ndarray:
        """
        __norm_rotation_frame(frame)

            Normalizes slit angle by rotating input frame

            Parameters
            ----------
            frame : np.ndarray
                hyperspectral frame

            Returns
            -------
                np.ndarray
        """
        angle = HSBuilder.__get_slit_angle(frame)
        #  rotate frame while angle is not in (-0.01; 0.01) degrees
        while abs(angle) > 0.01:
            h, w = frame.shape
            center_x, center_y = (w // 2, h // 2)
            angle = HSBuilder.__get_slit_angle(frame)
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            frame = cv2.warpAffine(frame, rotation_matrix, (w, h))

        return frame
    # ------------------------------------------------------------------------------------------------------------------

    # TODO implementation will move into device
    @staticmethod
    def __norm_barrel_distortion(frame: np.ndarray) -> np.ndarray:
        """
        __norm_barrel_distortion(frame)

            Removes barrel distortion on frame

            Parameters
            ----------
            frame : np.ndarray
                hyperspectral frame

            Returns
            -------
                np.ndarray
        """
        width = frame.shape[1]
        height = frame.shape[0]

        distCoeff = np.zeros((4, 1), np.float64)
        # TODO: replace by device features! IT'S HARDCODED
        k1 = -1.4e-5  # negative to remove barrel distortion
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0

        distCoeff[0, 0] = k1
        distCoeff[1, 0] = k2
        distCoeff[2, 0] = p1
        distCoeff[3, 0] = p2
        # assume unit matrix for camera
        cam = np.eye(3, dtype=np.float32)

        cam[0, 2] = width / 2.0  # define center x
        cam[1, 2] = height / 2.0  # define center y
        # TODO remake hardcoded values!
        cam[0, 0] = 10.  # define focal length x
        cam[1, 1] = 10.  # define focal length y

        # here the undistortion will be computed
        dst = cv2.undistort(frame, cam, distCoeff)
        return dst
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __norm_frame_camera_geometry(frame: np.ndarray,
                                     norm_rotation=False,
                                     barrel_dist_norm=False) -> np.ndarray:
        """
        Normalizes geometric distortions on frame:
            - rotation
            - barrel distortion

        Parameters
        ----------
        frame : np.ndarray

        Returns
        -------
        frame : np.ndarray

        """
        if norm_rotation:
            frame = HSBuilder.__norm_rotation_frame(frame=frame)
        if barrel_dist_norm:
            frame = HSBuilder.__norm_barrel_distortion(frame=frame)

        return frame
    # ------------------------------------------------------------------------------------------------------------------

    # TODO implementation will move into device
    @staticmethod
    def get_roi(frame: np.ndarray, roi_coords: Dict) -> np.ndarray:
        """
        For this moment works to microscope rough settings
        Parameters
        ----------
        frame :
        roi_coords:

        Returns
        -------

        """
        gap_coord = roi_coords['gap_coord']
        range_to_spectrum = roi_coords['range_to_spectrum']
        range_to_end_spectrum = roi_coords['range_to_end_spectrum']
        left_bound_spectrum = roi_coords['left_bound_spectrum']
        right_bound_spectrum = roi_coords['right_bound_spectrum']
        up_bound = gap_coord + range_to_spectrum
        down_bound = up_bound + range_to_end_spectrum

        return frame[up_bound: down_bound, left_bound_spectrum: right_bound_spectrum]
    # ------------------------------------------------------------------------------------------------------------------

    # TODO rename
    @staticmethod
    def __principal_slices(frame: np.ndarray,
                           nums_bands: int) -> np.ndarray:
        """
        __principal_slices(frame, nums_bands)

            Compresses the frame by number of channels

            Parameters
            ----------
            frame: np.ndarray
                2D frame which we wanna compress from shape (W, H) ---> (W, nums_bands)

            nums_bands: int
                Final numbers of channels

            Returns
            -------
                np.ndarray
        """
        n, m = frame.shape 

        width = m // nums_bands 
        gaussian_window = gaussian(width, width / 2.0, width / 6.0) 
        mid = len(gaussian_window) // 2 
        gaussian_window[mid] = 1.0 - np.sum(gaussian_window) + gaussian_window[mid] 
        ans = np.zeros(shape=(n, nums_bands), dtype=np.uint8)

        for j in range(nums_bands): 
            left_bound = j * m // nums_bands 
            ans[:, j] = np.tensordot(frame[:, left_bound:left_bound + len(gaussian_window)],
                                     gaussian_window,
                                     axes=([1], [0]))
        return ans
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def load_light_coeff(path_to_file):
        light_coeff = cv2.imread(path_to_file, 0)
        #light_coeff = HSBuilder.__norm_frame_camera_geometry(light_coeff,
        #                                                     norm_rotation=norm_rotation,
        #                                                     barrel_dist_norm=barrel_dist_norm)
        light_coeff = HSBuilder.get_roi(frame=light_coeff)
        return 1.0 / (light_coeff / np.max(light_coeff))
    # ------------------------------------------------------------------------------------------------------------------

    def build(self,
              principal_slices=False,
              norm_rotation=False,
              barrel_dist_norm=False,
              light_norm=False,
              roi=False,
              flip_wavelengths=False):
        """
            Creates HSI from device-data
        """

        #if light_norm:
        #    light_coeff = HSBuilder.load_light_coeff(path_to_file='./test_data/builder/micro_light_source.png')
        #else:
        #    light_coeff = None

        preproc_frames = []
        for frame in tqdm(self.frame_iterator,
                          total=len(self.frame_iterator),
                          desc='Preprocessing frames',
                          colour='blue'):
            frame = self.__norm_frame_camera_geometry(frame=frame,
                                                      norm_rotation=norm_rotation,
                                                      barrel_dist_norm=barrel_dist_norm)
            if roi:
                frame = HSBuilder.get_roi(frame=frame, roi_coords=self.roi_coords)
            if light_norm:
                frame = self.__norm_frame_camera_illumination(frame=frame, light_coeff=self.light_coeff)
            if principal_slices:
                frame = self.__principal_slices(frame.T, principal_slices)
            else:
                frame = frame.T
            preproc_frames.append(frame)
            
        data = np.array(preproc_frames)
        if self.path_to_gps:
            data = build_hypercube_by_videos(data.astype("uint8"), self.path_to_gps)
        if flip_wavelengths:
            data = np.flip(data, axis=2)
        self.hsi = HSImage(hsi=data, wavelengths=None)
    # ------------------------------------------------------------------------------------------------------------------

    def get_hsi(self) -> HSImage:
        """

        Returns
        -------
        self.hsi : HSImage
            Builded from source hsi object

        """
        try:
            return self.hsi
        except:
            pass
    # ------------------------------------------------------------------------------------------------------------------
