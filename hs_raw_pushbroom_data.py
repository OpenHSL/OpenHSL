class HSRawData:
    """
    Reader and iterator of raw data-files such as video,
    common images (png, bmp) and multichannel images (mat, gettiff, h5)

    """
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __len__(self):
        pass

    def __load_from_video(self):
        pass

    def __load_from_images(self):
        pass

    def __load_from_geotiff(self):
        pass

    def __load_from_mat(self):
        pass

    def __load_from_h5(self):
        pass

