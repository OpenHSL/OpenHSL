import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple


class RawImagesData:
    """
    RawImagesData(path_to_dir)
        Create iterator for images set.
        In each step return ndarray object
        Parameters
        ----------
        path_to_dir : str
            Path to directory with images
        Attributes
        ----------
        path_to_dir : str
        imgs_list : iterable
        Examples
        --------
        rid = RawImagesData("./some_directory")
        for frame in rid:
            some_operation(frame)
    """

    def __init__(self, path_to_dir: str):
        self.path_to_dir = path_to_dir
        self.imgs_list = iter(os.listdir(path_to_dir))

    def __iter__(self):
        return self

    def __next__(self):
        self.path_to_curr_img = next(self.imgs_list)
        path_to_img = f'{self.path_to_dir}/{self.path_to_curr_img}'
        #TODO maybe replace by cv2.imread?
        return np.array(Image.open(path_to_img).convert("L"))

    def __len__(self):
        return len(os.listdir(self.path_to_dir))


class RawVideoData:
    """
        RawVideoData(path_to_file_)

            #TODO make with cv2.video_capture

            Create iterator for videoframes.
            In each step return PIL.Image object # TODO maybe cv2 or np.array?

            Parameters
            ----------
            path_to_file : str
                Path to video file

            Attributes
            ----------
            path_to_file : str

            Examples
            --------
            rvd = RawVideoData("./some_path")

            for frame in rvd:
                some_operation(frame)

        """

    def __init__(self, path_to_file:str):
        self.path = path_to_file
        self.format = self.path.split(".")[-1]
        self.current_step = 0
        self.cap = cv2.VideoCapture(self.path)

    def __iter__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            self.frame = frame
            if ret:
                return self
            else:
                break

    def __len__(self):
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.length

    def __next__(self):
        self.current_step+=1
        frame = self.frame
        self.frame += 1
        if self.current_step < len(self):
            return frame
        else:
            raise StopIteration


class RawCsvData:
    """

    """
    def __init__(self, path_to_csv: str):
        self.path_to_csv = path_to_csv
        self.current_step = 0

        def return_name_of_video_from_df(df: pd.DataFrame, column: str) -> list:
            x = df.loc[(df['cam_ID'] == 'Hypercam start point')]
            x = x[column]
            return x.values

        def process_df(path_to_csv: str) -> Tuple[pd.DataFrame, list]:
            df = pd.read_csv(path_to_csv, sep=";")
            video_names = return_name_of_video_from_df(df, "timing")
            df = df[df["cam_ID"] == "Hypercam frame"]
            return df, video_names

        self.df, self.video_names = process_df(self.path_to_csv)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_step < len(self):
            self.current_step += 1
            return self.df.iloc[self.current_step-1]
        else:
            raise StopIteration
    
    def __len__(self):
        return len(self.df)


class RawMatData:

    def __init__(self, path_to_mat: str):
        self.path_to_mat = path_to_mat

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __len__(self):
        pass


class RawTiffData:
    # TODO may be realize with GDAL?
    def __init__(self, path_to_tiff: str):
        self.path_to_tiff = path_to_tiff

    def __iter__(self):
        return self

    def __next__(self):
        pass

    def __len__(self):
        pass



