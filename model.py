from abc import ABC, abstractmethod


class Model(ABC):
    """
    Model()

        Abstract class for decorating machine learning algorithms

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

    @abstractmethod
    def fit(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def predict(self):
        pass
    # ------------------------------------------------------------------------------------------------------------------
