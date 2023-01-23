import os
from PIL import Image


class RawImagesData:
    """
    RawImagesData(path_to_dir)

        Create iterator for images set.
        In each step return PIL.Image object

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
        return Image.open(path_to_img).convert("L")

    def __len__(self):
        return len(os.listdir(self.path_to_dir))


class RawVideoData:
    """
        RawVideoData(path_to_file)

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

    def __init__(self, path_to_file: str):
        self.path_to_file = path_to_file

    def __iter__(self):
        return self

    def __next__(self):
        pass

    def __len__(self):
        pass


class RawMatData:

    def __init__(self, path_to_mat: str):
        self.path_to_mat = path_to_mat

    def __iter__(self):
        return self

    def __next__(self):
        pass

    def __len__(self):
        pass


class RawTiffData:

    def __init__(self, path_to_tiff: str):
        self.path_to_tiff = path_to_tiff

    def __iter__(self):
        return self

    def __next__(self):
        pass

    def __len__(self):
        pass


class HSRawData:
    """
    HSRawData()

        Reader and iterator of raw data-files such as video,
        common images (png, bmp) and multichannel images (mat, gettiff, h5)

        Parameters
        ----------

        Attributes
        ----------

        See Also
        --------

        Notes
        -----

        Examples
        --------

    """
    def __init__(self, path_to_source, source_type):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def __iter__(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def __next__(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def _load_from_video(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def _load_from_images(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def _load_from_geotiff(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def _load_from_mat(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    def _load_from_h5(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------
