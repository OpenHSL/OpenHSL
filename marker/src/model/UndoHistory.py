class UndoHistory:
    def __init__(self, limit):
        self._undo_limit = limit
        self._undo_stack = []
        self._redo_stack = []

    def reset(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

    def has_undo(self):
        return len(self._undo_stack) > 0

    def has_redo(self):
        return len(self._redo_stack) > 0

    def save_state(self, state):
        self._undo_stack.insert(0, state)
        self._redo_stack.clear()
        while len(self._undo_stack) > self._undo_limit:
            self._undo_stack.pop(self._undo_limit)

    def undo(self, current_state):
        self._redo_stack.insert(0, current_state)
        return self._undo_stack.pop(0)

    def redo(self, current_state):
        self._undo_stack.insert(0, current_state)
        return self._redo_stack.pop(0)
