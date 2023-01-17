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

