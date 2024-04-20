from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QMouseEvent, QPainter
from PyQt6.QtWidgets import QApplication, QCheckBox, QGraphicsView, QHeaderView, QStyle, QStyleOptionButton, QWidget
from typing import List, Optional, Tuple


class CheckableHeaderView(QHeaderView):
    def __init__(self, orientation: Qt.Orientation, parent: QWidget = None):
        super().__init__(orientation, parent)

        self.check_list: List[bool] = []
        self.checkbox_list: List[QCheckBox] = []
        self.checkbox_rect_list: List[QRect] = []
        self.sectionCountChanged.connect(self.on_section_count_changed)
        self.checkbox_stylesheet = ""
        self.setMouseTracking(True)

    def get_check_list(self) -> List[bool]:
        return self.check_list

    def paintSection(self, painter: QPainter, rect: QRect, logical_index: int):
        painter.save()
        super().paintSection(painter, rect, logical_index)
        painter.restore()
        painter.translate(rect.topLeft())

        option = QStyleOptionButton()
        size = 13
        option.rect = QRect((29 - size) // 2, (rect.height() - size) // 2, size, size)
        option.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Active

        if self.check_list[logical_index]:
            option.state |= QStyle.StateFlag.State_On
        else:
            option.state |= QStyle.StateFlag.State_Off
        option.state |= QStyle.StateFlag.State_Off

        if self.checkbox_list[logical_index].property('hovered'):
            option.state |= QStyle.StateFlag.State_MouseOver

        self.style().drawControl(QStyle.ControlElement.CE_CheckBox, option, painter, self.checkbox_list[logical_index])
        # Track checkbox rect for mouse press event
        self.checkbox_rect_list[logical_index] = option.rect.translated(rect.topLeft().x(), rect.topLeft().y())

    def mouseMoveEvent(self, event: QMouseEvent):
        idx = self.logicalIndexAt(event.pos())
        cb = self.checkbox_list[idx]

        update_needed = self.checkbox_rect_list[idx].contains(event.pos()) != cb.property('hovered')
        if update_needed:
            cb.setProperty('hovered', self.checkbox_rect_list[idx].contains(event.pos()))
            self.updateSection(idx)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        idx = self.logicalIndexAt(event.pos())

        if self.checkbox_rect_list[idx].contains(event.pos()):
            self.check_list[idx] = not self.check_list[idx]
            self.updateSection(idx)
        super().mousePressEvent(event)

    @pyqtSlot(int, int)
    def on_section_count_changed(self, old_count: int, new_count: int):
        if new_count > old_count:
            for i in range(new_count - old_count):
                self.check_list.append(False)
                cb = QCheckBox()
                # Workaround
                cb.setStyleSheet(self.checkbox_stylesheet)
                cb.setProperty('hovered', False)

                self.checkbox_list.append(cb)
                self.checkbox_rect_list.append(cb.rect())
        else:
            self.check_list = self.check_list[0:new_count]
            self.checkbox_list = self.checkbox_list[0:new_count]
            self.checkbox_rect_list = self.checkbox_rect_list[0:new_count]
