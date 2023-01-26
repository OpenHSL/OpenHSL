import os
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
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
        RawVideoData(path_to_file: str)

            Create iterator for videoframes.
            In each step return np.array object.

            Parameters
            ----------
            path_to_file : str
                Path to video file

            Attributes
            ----------
            path : pathlib.Path
                Path to video file
            format: str
                Video format
            current_step : int
                Current step of iteration
            cap : cv2.VideoCapture
                Video capture object
            frame: np.array
                Current frame
            length: int
                Count of videoframes

            Examples
            --------
            rvd = RawVideoData("./some_path")

            len(rvd)
            for frame in rvd:
                some_operation(frame)
            
        """

    def __init__(self, path_to_file: str):
        self.path = Path(path_to_file)
        self.format = self.path.suffix
        self.current_step = 0
        self.cap = cv2.VideoCapture(str(self.path))

    def __iter__(self):
        return self

    def __len__(self):
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.length

    def __next__(self):
        while self.cap.isOpened():
            if self.current_step < len(self):
                self.current_step += 1
                ret, self.frame = self.cap.read()
                if ret:
                    return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                else:
                    raise StopIteration
            else:
                raise StopIteration


class RawCsvData:
    """
        RawCsvData(path_to_csv: str, video_name: str)

            Create iterator for csv lines.
            In each step return dict object.

            Parameters
            ----------
            path_to_csv : str
                Path to csv file

            Attributes
            ----------
            path_to_csv : pathlib.Path
                Path to csv file
            video_name : str
                Name of video file in csv file
            current_step : int
                Current step of iteration
            df : pd.DataFrame
                Dataframe for video with csv data

            Examples
            --------
            video_name = "rec_2021.avi"
            rvd = RawCsvData(path_to_csv="./some_path.csv", video_name=video_name)

            len(rcd)
            for line in rcd:
                some_operation(line)
            
    """
    def __init__(self, path_to_csv: str, video_name: str):
        self.video_name = str(video_name).split("_")[-1].split(".")[0]
        self.current_step = 0

        def if_video_not_in_df(df: pd.DataFrame, video_name: str):
            x = df.loc[(df['cam_ID'] == 'Hypercam start point')]
            x = x["timing"]
            if video_name not in x.values:
                raise ValueError(f"Video with name {video_name} not found in {path_to_csv} file")

        def process_df(path_to_csv: str) -> Tuple[pd.DataFrame, list]:
            df = pd.read_csv(path_to_csv, sep=";")
            if_video_not_in_df(df, self.video_name)
            df = df.fillna(method="ffill")
            df = df[df["timing"] == self.video_name]
            df = df[df["cam_ID"] == "Hypercam frame"]
            return df

        self.df = process_df(path_to_csv)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_step < len(self):
            self.current_step += 1
            return dict(self.df.iloc[self.current_step-1])
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



