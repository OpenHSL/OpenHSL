import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from typing import Tuple

from utils import dir_exists, load_data

SUPPORTED_VIDEO_FORMATS = ("mp4", "avi")
SUPPORTED_IMG_FORMATS = ("jpg", "png", "bmp")
SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS + SUPPORTED_IMG_FORMATS

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

    def __init__(self, path_to_files: list):
        self.path_to_files = path_to_files

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self):
        if self.current_step < len(self):
            img = np.array(Image.open(self.path_to_files[self.current_step]).convert("L"))
            self.current_step += 1
            return img
        else:
            raise StopIteration

    def __len__(self):
        return len(self.path_to_files)


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

    def __init__(self, path_to_files: list):
        self.paths = path_to_files
        self.caps = [cv2.VideoCapture(file) for file in self.paths]

    def __iter__(self):
        self.current_step = 0
        return self

    def __len__(self):
        self.length = 0
        for cap in self.caps:
            self.length += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.length

    def __next__(self):
        if self.current_step < len(self):
            self.current_step += 1
            #read in grayscale
            for cap in self.caps:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return frame
        else:
            raise StopIteration
    
class RawData:
    """

    """
    def __init__(self, path_to_data: str, type_data: str):
        if path_to_data.split(".")[-1] not in SUPPORTED_FORMATS:
            if not dir_exists(path_to_data):
                raise ValueError("Path {} not found or {} format not supported".format(path_to_data, path_to_data.split(".")[-1]))
            if type_data == "images":
                files = load_data(path_to_data, SUPPORTED_IMG_FORMATS)
            elif type_data == "video":
                files = load_data(path_to_data, SUPPORTED_VIDEO_FORMATS)
        
        else:
            files = [path_to_data]

        if type_data == "images":
            self.raw_data = RawImagesData(path_to_files=files)

        elif type_data == "video":
            self.raw_data = RawVideoData(path_to_files=files)

        else:
            raise ValueError("type_data must be 'images' or 'video'")

    def __iter__(self):
        return self.raw_data.__iter__()

    def __next__(self):
        return next(self.raw_data)

    def __len__(self):
        self.raw_data.__len__()


class RawCsvData:
    """
        RawCsvData(path_to_csv: str, video_name: str)

            Create iterator for csv lines.
            In each step return dict object.
            for filename in load_data(dir_input, ".avi"): #filename = dir_input + 
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
        Path(path_to_csv)
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



