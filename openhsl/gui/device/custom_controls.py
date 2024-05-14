import matplotlib as mpl
from PyQt6.QtCore import QAbstractTableModel, QItemSelection, QModelIndex, QObject, QPoint, QPointF, QRect, QRectF, \
    Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QCheckBox, QGraphicsView, QHeaderView, QStyle, QStyleOptionButton, QWidget
from typing import Any, List, Optional, Tuple
import openhsl.gui.device.utils as hsd_gui_utils


class EquationParamsTableHeaderViewHorizontal(QHeaderView):
    def __init__(self, orientation: Qt.Orientation, parent: QWidget = None):
        super().__init__(orientation, parent)
        self.pixmaps: List[QPixmap] = []
        self.pixmaps_selected: List[QPixmap] = []
        self.pixmap_color = '#d0d0d0'
        self.pixmap_selected_color = '#d0d0d0'
        self.font_size = None
        self.sectionCountChanged.connect(self.on_section_count_changed)

    def clear_data(self):
        self.pixmaps.clear()
        self.pixmaps_selected.clear()

    def paintSection(self, painter: QPainter, rect: QRect, logical_index: int):
        painter.save()
        super().paintSection(painter, rect, logical_index)
        painter.restore()
        painter.translate(rect.topLeft())

        if logical_index < len(self.pixmaps):
            pixmap = self.pixmaps[logical_index]

            # Selected sections
            indexes = self.selectionModel().selectedIndexes()
            if len(indexes) > 0:
                indexes = set([m.column() for m in indexes])
                if logical_index in indexes:
                    pixmap = self.pixmaps_selected[logical_index]

            xpix = (rect.width() - pixmap.size().width()) // 2
            ypix = (rect.height() - pixmap.size().height()) // 2
            rect = QRect(xpix, ypix, pixmap.size().width(),
                         pixmap.size().height())
            painter.drawPixmap(rect, pixmap)
            self.setMinimumWidth(pixmap.size().width() + 20)

    def update_labels(self, old_count: int, new_count: int):
        for i in range(old_count, old_count + new_count):
            pixmap = QPixmap(f'icons_gen:bdet-header-hor-{i}.png')
            self.pixmaps.append(pixmap)
            pixmap_selected = QPixmap(f'icons_gen:bdet-header-hor-selected-{i}.png')
            self.pixmaps_selected.append(pixmap_selected)

    @pyqtSlot(int, int)
    def on_section_count_changed(self, old_count: int, new_count: int):
        if new_count > old_count:
            for i in range(old_count, old_count + new_count):
                pixmap = QPixmap(f'icons_gen:bdet-header-hor-{i}.png')
                self.pixmaps.append(pixmap)
                pixmap_selected = QPixmap(f'icons_gen:bdet-header-hor-selected-{i}.png')
                self.pixmaps_selected.append(pixmap_selected)
        else:
            self.pixmaps = self.pixmaps[0:new_count]
            self.pixmaps_selected = self.pixmaps_selected[0:new_count]


class EquationParamsTableHeaderViewVertical(QHeaderView):
    checked_section_count_changed = pyqtSignal()

    def __init__(self, orientation: Qt.Orientation, parent: QWidget = None):
        super().__init__(orientation, parent)

        self.check_list: List[bool] = []
        self.checkbox_list: List[QCheckBox] = []
        self.checkbox_rect_list: List[QRect] = []
        self.pixmaps: List[QPixmap] = []
        self.pixmaps_selected: List[QPixmap] = []
        self.pixmap_color = '#d0d0d0'
        self.pixmap_selected_color = '#d0d0d0'
        self.font_size = None
        self.sectionCountChanged.connect(self.on_section_count_changed)
        self.checkbox_stylesheet = ""
        self.setMouseTracking(True)

    def clear_data(self):
        self.check_list.clear()
        self.checkbox_list.clear()
        self.checkbox_rect_list.clear()
        self.pixmaps.clear()
        self.pixmaps_selected.clear()
        self.checked_section_count_changed.emit()

    def get_check_list(self) -> List[bool]:
        return self.check_list

    def set_section_checked(self, idx: int, value: bool):
        if idx < len(self.check_list):
            self.check_list[idx] = value
            self.updateSection(idx)
            self.checked_section_count_changed.emit()

    def paintSection(self, painter: QPainter, rect: QRect, logical_index: int):
        painter.save()
        super().paintSection(painter, rect, logical_index)
        painter.restore()
        painter.translate(rect.topLeft())

        if logical_index < len(self.check_list):
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

            pixmap = self.pixmaps[logical_index]

            # Selected sections
            indexes = self.selectionModel().selectedIndexes()
            if len(indexes) > 0:
                indexes = set([m.row() for m in indexes])
                if logical_index in indexes:
                    pixmap = self.pixmaps_selected[logical_index]

            xpix = (rect.width() - pixmap.size().width()) // 2 + rect.x()
            ypix = (rect.height() - pixmap.size().height()) // 2
            rect = QRect(xpix, ypix, pixmap.size().width(),
                         pixmap.size().height())
            painter.drawPixmap(rect.translated(option.rect.topLeft().x(), 0), pixmap)
            self.setMinimumWidth(pixmap.size().width() + size + 16 + 20)

    def mouseMoveEvent(self, event: QMouseEvent):
        idx = self.logicalIndexAt(event.pos())
        if idx < len(self.checkbox_list):
            cb = self.checkbox_list[idx]

            update_needed = self.checkbox_rect_list[idx].contains(event.pos()) != cb.property('hovered')
            if update_needed:
                cb.setProperty('hovered', self.checkbox_rect_list[idx].contains(event.pos()))
                self.updateSection(idx)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        idx = self.logicalIndexAt(event.pos())

        if idx < len(self.checkbox_list):
            if self.checkbox_rect_list[idx].contains(event.pos()):
                self.check_list[idx] = not self.check_list[idx]
                self.updateSection(idx)
                self.checked_section_count_changed.emit()
        super().mousePressEvent(event)

    @pyqtSlot(int, int)
    def on_section_count_changed(self, old_count: int, new_count: int):
        if new_count > old_count:
            for i in range(old_count, old_count + new_count):
                pixmap = QPixmap(f'icons_gen:bdet-header-vert-{i}.png')
                self.pixmaps.append(pixmap)
                pixmap_selected = QPixmap(f'icons_gen:bdet-header-vert-selected-{i}.png')
                self.pixmaps_selected.append(pixmap_selected)
            for i in range(old_count, old_count + new_count):
                self.check_list.append(False)
                cb = QCheckBox()
                # Workaround
                cb.setStyleSheet(self.checkbox_stylesheet)
                cb.setProperty('hovered', False)
                self.checkbox_list.append(cb)
                self.checkbox_rect_list.append(QRect())
        else:
            self.check_list = self.check_list[0:new_count]
            self.checkbox_list = self.checkbox_list[0:new_count]
            self.checkbox_rect_list = self.checkbox_rect_list[0:new_count]
            self.pixmaps = self.pixmaps[0:new_count]
            self.pixmaps_selected = self.pixmaps_selected[0:new_count]
        self.checked_section_count_changed.emit()


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
        elif event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton):
            return None
        else:
            return super(HSGraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            pos_on_scene, out_of_bound = self.get_pos_on_scene(event.pos())
            self.marquee_area_bottom_right = pos_on_scene
            self.marquee_area_changed.emit(self.marquee_area_top_left, self.marquee_area_bottom_right)
        else:
            return super(HSGraphicsView, self).mouseReleaseEvent(event)


class EquationParamsTableItem:
    def __init__(self, power: int = 0, coeff: float = 1.0, factor: int = 1):
        self.power = power
        self.coeff = coeff
        self.factor = factor

    @staticmethod
    def from_list(cls, params: List):
        obj = cls()
        if len(params) == 3:
            obj.power = params[0]
            obj.coeff = params[1]
            obj.factor = params[2]
        return obj

    def get_item(self):
        return {'power': self.power, 'coeff': self.coeff, 'factor': self.factor}

    def get_item_data(self, index: int):
        value = None

        if index == 0:
            value = self.power
        elif index == 1:
            value = self.coeff
        elif index == 1:
            value = self.factor

        return value

    def set_item(self, power: int, coeff: float = 1.0, factor: int = 1):
        self.power = power
        self.coeff = coeff
        self.factor = factor

    def set_item_data(self, index: int, value):
        if index == 0:
            self.power = value
        elif index == 1:
            self.coeff = value
        elif index == 2:
            self.factor = value

    def to_list(self):
        return [self.power, self.coeff, self.factor]


class EquationParamsTableModel(QAbstractTableModel):
    def __init__(self, parent: QObject = None):
        super(EquationParamsTableModel, self).__init__(parent)
        self.items: List[EquationParamsTableItem] = []
        self.horizontal_header_labels = ['Coefficient', 'Factor, 10^x']

    def clear(self):
        self.beginRemoveRows(QModelIndex(), 0, self.rowCount() - 1)
        self.items.clear()
        self.endRemoveRows()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row_index = index.row()
        col_index = index.column()

        v = self.items[row_index].to_list()[col_index + 1]

        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        elif role in [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]:
            return v
        else:
            return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        return super(EquationParamsTableModel, self).flags(index) | Qt.ItemFlag.ItemIsEditable

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            return None

    def insertRow(self, row: int, parent: QModelIndex = QModelIndex()) -> bool:
        if row < 0:
            row = 0

        self.beginInsertRows(parent, row, row)

        item = EquationParamsTableItem()
        self.items.insert(row, item)

        self.endInsertRows()

        return True

    def insertRows(self, row: int, count: int, parent: QModelIndex = QModelIndex()) -> bool:
        if row < 0:
            row = 0

        self.beginInsertRows(parent, row, row + count - 1)

        for i in range(count):
            item = EquationParamsTableItem()
            self.items.insert(row + i, item)

        self.endInsertRows()

        return True

    def load_data_from_list(self, data: List[List], row_count: int):
        if len(self.items) > 0:
            self.clear()
        self.insertRows(0, row_count)
        for i in range(len(data[0])):
            for j, item_value in enumerate(data[1:]):
                self.setData(self.index(int(data[0][i]), j), item_value[i])

    def removeRow(self, row: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginRemoveRows(parent, row, row)

        del self.items[row]

        self.endRemoveRows()

        return True

    def removeRows(self, row: int, count: int, parent: QModelIndex = QModelIndex()) -> bool:
        self.beginRemoveRows(parent, row, row + count - 1)

        remove_indices = set(range(row, row + count))
        self.items = [i for j, i in enumerate(self.items) if j not in remove_indices]

        self.endRemoveRows()

        return True

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.items)

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False

        row_index = index.row()
        col_index = index.column()

        if role == Qt.ItemDataRole.EditRole:
            self.items[row_index].set_item_data(col_index + 1, value)
            self.dataChanged.emit(index, index)
            return True

        return False
