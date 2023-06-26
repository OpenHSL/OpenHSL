class ObservableSubject:
    def __init__(self):
        self.observer_callbacks = []

    def attach(self, callback):
        self.observer_callbacks.append(callback)

    def detach(self, callback):
        self.observer_callbacks.remove(callback)

    def notify(self):
        for callback in self.observer_callbacks:
            callback()
