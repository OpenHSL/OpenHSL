import os

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
        self.path_to_source = path_to_source
        self.source_type = source_type
    # ------------------------------------------------------------------------------------------------------------------

    def __iter__(self):
        if self.source_type == 'images':
            list_imgs = os.listdir(self.path_to_source)
            self.current_index = 0
        if self.source_type == 'video':
            ...
        return self
    # ------------------------------------------------------------------------------------------------------------------

    def __next__(self):
        tmp = self.current_index
        self.current_index += 1
        return
        raise StopIteration
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
