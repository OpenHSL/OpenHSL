class KeyboardModel:
    def __init__(self, master):
        self.master = master
        self._pressed = set()

        self.master.bind("<Key>", self._on_press)
        self.master.bind("<KeyRelease>", self._on_release)

    def is_pressed(self, keysym):
        return keysym in self._pressed

    def _on_press(self, e):
        self._pressed.add(e.keysym)

    def _on_release(self, e):
        if e.keysym in self._pressed:
            self._pressed.remove(e.keysym)


