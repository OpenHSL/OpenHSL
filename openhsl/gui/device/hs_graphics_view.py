from PyQt6.QtCore import QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QGraphicsView, QWidget
from typing import Optional


class HSGraphicsView(QGraphicsView):
    marquee_area_changed = pyqtSignal(QPoint, QPoint)
    def __init__(self, parent: Optional[QWidget] = None):
        super(HSGraphicsView, self).__init__(parent)
        self.marquee_area_top_left = QPoint()
        self.marquee_area_bottom_right = QPoint()

    def mousePressEvent(self, event):
        if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
            if event.button() == Qt.MouseButton.LeftButton:
                self.marquee_area_top_left = event.pos()
                self.marquee_area_changed.emit(self.marquee_area_top_left, self.marquee_area_bottom_right)
        else:
            return super(HSGraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.marquee_area_bottom_right = event.pos()
                self.marquee_area_changed.emit(self.marquee_area_top_left, self.marquee_area_bottom_right)
        else:
            return super(HSGraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.marquee_area_bottom_right = event.pos()
            self.marquee_area_changed.emit(self.marquee_area_top_left, self.marquee_area_bottom_right)
        else:
            return super(HSGraphicsView, self).mouseReleaseEvent(event)