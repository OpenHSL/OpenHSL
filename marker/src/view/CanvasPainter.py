from tkinter import *
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np


class CanvasPainter:
    def __init__(self, canvas, model):
        self.canvas = canvas
        self.model = model
        self.model.subject.project.attach(self._update_project)
        self.model.subject.layer.attach(self._update_layer)

        # constants
        self._color_paint = 'white'
        self._color_erase = 'black'

        # vars needing reset
        self._bg_comp = None
        self._fg_comp = None
        self._brush_position = None
        self._pan_position = None

    def _update_project(self):
        if self.model.isProjectLoaded:
            if self.model.project.backgroundImagePath:
                self._bg_image = Image.open(self.model.project.backgroundImagePath).convert('RGBA')
            self._update_layer()
        else:
            self.canvas.delete(ALL)

    def _update_layer(self):
        self._bg_comp = self._comp_bg_image()
        if self.model.project.maskOpaque:
            self._fg_comp = self._comp_fg_image()
        self.render_canvas_image()

    def render_canvas_image(self):
        active_layer_image = self._get_masked_image(self.model.project.activeMask)
        composite = Image.alpha_composite(self._bg_comp, active_layer_image)
        if self.model.project.maskOpaque:
            composite = Image.alpha_composite(composite, self._fg_comp)
        else:
            # if masks are transparent, all layers in front of the active layer need to be re-rendered
            #   immediately after the active layer is re-rendered
            #   and ignore the _fg_comp since that only works for opaque masks
            for i in range(self.model.project.activeMask + 1, self.model.project.numMasks):
                layer_image = self._get_masked_image(i)
                composite = Image.alpha_composite(composite, layer_image)

        self.canvas.delete(ALL)

        # convert the PIL image to TK PhotoImage
        # set the canvas.image property, it wont work without this step
        self.canvas.image = ImageTk.PhotoImage(composite)
        self.canvas.create_image(self.model.canvas.cs_crop_x, self.model.canvas.cs_crop_y,
                                 image=self.canvas.image, anchor=NW)

        # for debugging
        # self.render_camera_outline()

    def render_brush_outline(self, x, y):
        r = self.model.brushSize * self.model.canvas.zoom
        self.canvas.create_oval(x - r, y - r, x + r, y + r)

    def render_camera_outline(self):
        # for debugging zoom, render camera center
        x = self.model.canvas.canvas_w // 2
        y = self.model.canvas.canvas_h // 2
        r = self.model.brushSize * self.model.canvas.zoom
        self.canvas.create_rectangle(x - r, y - r, x + r, y + r)

    def export_comp_image(self):
        old_zoom = self.model.canvas.zoom
        self.model.canvas.set_zoom(1)

        temp_bg = self._comp_bg_image(show_all=True)
        temp_fg = self._comp_fg_image(show_all=True)
        temp_active_layer_image = self._get_masked_image(self.model.project.activeMask, show_all=True)
        composite = Image.alpha_composite(temp_bg, temp_active_layer_image)
        composite = Image.alpha_composite(composite, temp_fg)

        # send the PIL image to the model to save to disk
        self.model.export_comp_image(composite)
        self.model.canvas.set_zoom(old_zoom)

    def zoom(self, e):
        self._update_project()
        self.render_brush_outline(e.x, e.y)

    def resize(self):
        self._update_layer()

    def pan(self, e):
        if self._pan_position:
            old_x, old_y = self._pan_position
            dx = old_x - e.x
            dy = old_y - e.y
            self.model.canvas.move_camera_position(dx, dy)
            self._update_layer()
        self._pan_position = (e.x, e.y)

    def end_pan(self, _):
        self._pan_position = None

    def paint(self, e):
        self._edit_active_mask(e, (255, 255, 255))

    def erase(self, e):
        self._edit_active_mask(e, (0, 0, 0))

    def end_brush_stroke(self, _):
        self._brush_position = None
        if self.model.isCurrentSaved:
            self.model.set_mask_edited()

    ###########################################################################
    #
    #  helpers
    #
    ###########################################################################

    def _edit_active_mask(self, e, color):
        active_layer = self.model.project.activeLayer
        x, y = self.model.canvas.mouse_canvas_to_world(e)
        x = x // self.model.canvas.zoom
        y = y // self.model.canvas.zoom

        if active_layer.isVisible and not active_layer.isLocked:
            if self._brush_position:
                bx, by = self._brush_position
                cv.line(active_layer.cvMask, (bx, by), (x, y), color, self.model.brushSize * 2)
            else:
                self.model.save_undo_state()
            cv.circle(active_layer.cvMask, (x, y), self.model.brushSize, color, -1)

        self.render_canvas_image()
        self.render_brush_outline(e.x, e.y)
        self._brush_position = (x, y)

    def _get_zoom_cv(self, cv_image):
        zoom_img = cv_image[
                   self.model.canvas.ms_zoom_top:self.model.canvas.ms_zoom_bottom,
                   self.model.canvas.ms_zoom_left:self.model.canvas.ms_zoom_right,
                   :
        ]
        zoom = self.model.canvas.zoom
        if zoom > 1:
            zoom_img = np.kron(zoom_img, np.ones((zoom, zoom, 1), dtype=np.uint8))
        return zoom_img

    def _get_zoom_cv_bg_image(self):
        ws_zoom_size = self.model.canvas.ws_zoom_size()
        cv_bg = self.model.project.cvBackgroundImage
        zoom_cv_bg = self._get_zoom_cv(cv_bg)
        image = Image.fromarray(zoom_cv_bg)
        image.resize(ws_zoom_size)
        return image

    def _get_mask(self, cv_mask):
        zoom_img = cv_mask[
                   self.model.canvas.ms_zoom_top:self.model.canvas.ms_zoom_bottom,
                   self.model.canvas.ms_zoom_left:self.model.canvas.ms_zoom_right
        ]
        zoom = self.model.canvas.zoom
        if zoom > 1:
            zoom_img = np.kron(zoom_img, np.ones((zoom, zoom), dtype=np.uint8))
        return zoom_img

    def _get_masked_image(self, mask_num, show_all=False):
        ws_zoom_size = self.model.canvas.ws_zoom_size()
        if mask_num < self.model.project.numMasks:
            layer = self.model.project.get_layer_by_z(mask_num)
            if self.model.project.maskOpaque:
                mask = Image.fromarray(self._get_mask(layer.cvMask))
            else:
                mask = Image.fromarray(self._get_mask(layer.cvMask) // 2)
            mask.convert('L').resize(ws_zoom_size)
            image = Image.new('RGBA', ws_zoom_size, layer.color)
            if layer.isVisible or show_all:
                image.putalpha(mask)
            else:
                image.putalpha(0)
            return image
        else:
            image = Image.new('RGBA', ws_zoom_size, 'magenta')
            image.putalpha(0)
            return image

    def _comp_bg_image(self, show_all=False):
        if self.model.project.backgroundImagePath:
            composite = self._get_zoom_cv_bg_image()
        else:
            composite = Image.new('RGBA', self.model.canvas.ws_zoom_size(), self.model.project.default_background_color)
        for i in range(self.model.project.activeMask):
            front = self._get_masked_image(i, show_all)
            composite = Image.alpha_composite(composite, front)
        return composite

    def _comp_fg_image(self, show_all=False):
        composite = self._get_masked_image(self.model.project.activeMask + 1, show_all)
        for i in range(self.model.project.activeMask + 1, self.model.project.numMasks):
            front = self._get_masked_image(i, show_all)
            composite = Image.alpha_composite(composite, front)
        return composite
