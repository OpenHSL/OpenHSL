from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QGraphicsView, QWidget
from typing import Optional, Tuple


class HSGraphicsView(QGraphicsView):
    marquee_area_changed = pyqtSignal(QPointF, QPointF)

    def __init__(self, parent: Optional[QWidget] = None):
        super(HSGraphicsView, self).__init__(parent)
        self.marquee_area_top_left = QPoint()
        self.marquee_area_bottom_right = QPoint()

    def get_pos_on_scene(self, event_pos: QPoint) -> Tuple[QPointF, bool]:
        scene_rect: QRectF = self.scene().sceneRect()
        pos_on_scene = self.mapToScene(event_pos)
        scene_rect_top_left: QPointF = scene_rect.topLeft()
        scene_rect_bottom_right: QPointF = scene_rect.bottomRight()

        out_of_bound = False

        if pos_on_scene.x() < scene_rect_top_left.x():
            pos_on_scene.setX(scene_rect_top_left.x() + 1)
        if pos_on_scene.y() < scene_rect_top_left.y():
            pos_on_scene.setY(scene_rect_top_left.y() + 1)
        if pos_on_scene.x() > scene_rect_bottom_right.x():
            pos_on_scene.setX(scene_rect_bottom_right.x() - 1)
        if pos_on_scene.y() > scene_rect_bottom_right.y():
            pos_on_scene.setY(scene_rect_bottom_right.y() - 1)

        if pos_on_scene != self.mapToScene(event_pos):
            out_of_bound = True

        return pos_on_scene, out_of_bound

    def mousePressEvent(self, event):
        if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
            if event.button() == Qt.MouseButton.LeftButton:
                pos_on_scene, out_of_bound = self.get_pos_on_scene(event.pos())
                self.marquee_area_top_left = pos_on_scene
                self.marquee_area_bottom_right = pos_on_scene
                self.marquee_area_changed.emit(self.marquee_area_top_left, self.marquee_area_bottom_right)
        else:
            return super(HSGraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if Qt.KeyboardModifier.ControlModifier == QApplication.keyboardModifiers():
            if event.buttons() & Qt.MouseButton.LeftButton:
                pos_on_scene, out_of_bound = self.get_pos_on_scene(event.pos())
                self.marquee_area_bottom_right = pos_on_scene
                self.marquee_area_changed.emit(self.marquee_area_top_left, self.marquee_area_bottom_right)
        else:
            return super(HSGraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            pos_on_scene, out_of_bound = self.get_pos_on_scene(event.pos())
            self.marquee_area_bottom_right = pos_on_scene
            self.marquee_area_changed.emit(self.marquee_area_top_left, self.marquee_area_bottom_right)
        else:
            return super(HSGraphicsView, self).mouseReleaseEvent(event)
