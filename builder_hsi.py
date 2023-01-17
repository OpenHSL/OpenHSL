from HSI import HSImage

class builder:

    def __init__(self, raw_hsi):
        self.hsi = []
        for layer in raw_hsi:
            self.hsi.append(layer)

    def some_preparation_on_hsi(self):
        pass

    def get_hsi(self) -> HSImage:
        try:
            return HSImage(self.hsi)
        except:
            pass
