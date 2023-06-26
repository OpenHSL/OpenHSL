"""
There are 3 different spaces: mask space, world space, and canvas space.
- mask space corresponds to the project mask images
- world space corresponds to mask space, but scaled up by the zoom factor
- canvas space corresponds to the tkinter canvas, which can be any arbitrary size due to resizing the window

- canvas space and world space are at the same scale, the zoom factor
- zoom size is the smallest rectangle that is the canvas size or larger, where both axes are a multiple of zoom factor
    world space zoom is exactly zoom factor times larger than mask space zoom

- on load:
    mask img size is set and it never changes until unload
    zoom factor is initialized to 1
    canvas size is read from the tkinter canvas
    all other vars are derived from these 3 vars

- on zoom change:
    world size and camera position are recalculated
    clamp the camera since it may be too close to the edge
    all other vars are derived again

- on camera position change (pan):
    move the camera by relative change between consecutive mouse positions
    clamp the camera since the user can drag very far away
    all other vars are derived again

- on canvas size change (resize):
    derive the new world space crop based on the new canvas size and the old camera position
    clamp the camera since it may be too close to the edge if the canvas size decreased
    re-derive the world space crop based on the new canvas size and the new camera position
    all other vars are derived again

"""

import math

from src.Utils import Utils


class CanvasModel:
    def __init__(self, is_reset=False):
        if not is_reset:
            # constants
            self.min_zoom = 1
            self.max_zoom = 8

        # zoom vars
        self.zoom = 1

        # mask img vars
        self.mask_w = 0
        self.mask_h = 0

        # derived from zoom and mask img size
        self.world_w = 0
        self.world_h = 0

        #############################################

        # canvas vars
        self.canvas_w = 0
        self.canvas_h = 0

        # derived from zoom, mask img size, canvas size

        # world space vars
        self.ws_camera_x = 0
        self.ws_camera_y = 0
        self.ws_crop_w = 0
        self.ws_crop_h = 0
        self.ws_crop_x = 0
        self.ws_crop_y = 0
        self.ws_canvas_x = 0
        self.ws_canvas_y = 0

        self.ws_zoom_left = 0
        self.ws_zoom_top = 0
        self.ws_zoom_right = 0
        self.ws_zoom_bottom = 0

        # mask space vars
        self.ms_zoom_left = 0
        self.ms_zoom_top = 0
        self.ms_zoom_right = 0
        self.ms_zoom_bottom = 0

        # canvas space vars
        self.cs_crop_x = 0
        self.cs_crop_y = 0

        #############################################

    def reset(self):
        self.__init__(True)

    def ws_zoom_size(self):
        return self.ws_zoom_right - self.ws_zoom_left, self.ws_zoom_bottom - self.ws_zoom_top

    def ms_zoom_size(self):
        return self.ms_zoom_right - self.ms_zoom_left, self.ms_zoom_bottom - self.ms_zoom_top

    def load(self, img_size, canvas_size):
        self.reset()

        self.canvas_w, self.canvas_h = canvas_size
        self.mask_w, self.mask_h = img_size
        self.world_w = self.mask_w * self.zoom
        self.world_h = self.mask_h * self.zoom
        self.ws_camera_x = self.world_w // 2
        self.ws_camera_y = self.world_h // 2

        self._derive_space_vars()

    def set_zoom(self, zoom):
        old_zoom = self.zoom
        self.zoom = Utils.clamp(zoom, self.min_zoom, self.max_zoom)
        
        # manually derive world and camera, since old camera position should only be scaled during zoom
        delta_zoom = 1.0 * self.zoom / old_zoom
        self.world_w = self.mask_w * self.zoom
        self.world_h = self.mask_h * self.zoom
        self.ws_camera_x = int(math.floor(self.ws_camera_x * delta_zoom))
        self.ws_camera_y = int(math.floor(self.ws_camera_y * delta_zoom))

        self._clamp_camera()
        self._derive_space_vars()

    def set_mouse_zoom(self, zoom, e):
        x, y = self.mouse_canvas_to_world(e)

        # move camera position with no clamping and no deriving space vars
        dx = x - self.ws_camera_x
        dy = y - self.ws_camera_y
        self.ws_camera_x += dx
        self.ws_camera_y += dy

        # zoom with no clamping and no deriving space vars
        old_zoom = self.zoom
        self.zoom = Utils.clamp(zoom, self.min_zoom, self.max_zoom)

        # manually derive world and camera, since old camera position should only be scaled during zoom
        delta_zoom = 1.0 * self.zoom / old_zoom
        self.world_w = self.mask_w * self.zoom
        self.world_h = self.mask_h * self.zoom
        self.ws_camera_x = int(math.floor(self.ws_camera_x * delta_zoom))
        self.ws_camera_y = int(math.floor(self.ws_camera_y * delta_zoom))

        # move the world so the new camera moves to the mouse
        # clamp and derive space vars
        self.move_camera_position(-dx, -dy)

    def move_camera_position(self, dx, dy):
        self.ws_camera_x += dx
        self.ws_camera_y += dy
        self._clamp_camera()
        self._derive_space_vars()

    def resize_canvas(self, canvas_size):
        self.canvas_w, self.canvas_h = canvas_size

        self._derive_ws_crop()
        self._clamp_camera()
        self._derive_space_vars()

    def mouse_canvas_to_world(self, e):
        # need no offset if canvas is larger than world
        # need positive offset if canvas is smaller than world
        x_offset = max(0, self.ws_canvas_x)
        y_offset = max(0, self.ws_canvas_y)
        x = (e.x - self.cs_crop_x + x_offset)
        y = (e.y - self.cs_crop_y + y_offset)
        return x, y

    def mouse_canvas_to_mask(self, e):
        x, y = self.mouse_canvas_to_world(e)
        return x // self.zoom, y // self.zoom

    ###########################################################################
    #
    #  helpers
    #
    ###########################################################################

    def _clamp_camera(self):
        left = self.ws_crop_w // 2
        right = self.world_w - int(math.ceil(self.ws_crop_w / 2.0))
        self.ws_camera_x = Utils.clamp(self.ws_camera_x, left, right)

        top = self.ws_crop_h // 2
        bottom = self.world_h - int(math.ceil(self.ws_crop_h / 2.0))
        self.ws_camera_y = Utils.clamp(self.ws_camera_y, top, bottom)

    def _derive_ws_crop(self):
        # world space vars
        self.ws_crop_w = min(self.world_w, self.canvas_w)
        self.ws_crop_h = min(self.world_h, self.canvas_h)
        self.ws_crop_x = self.ws_camera_x - self.ws_crop_w // 2
        self.ws_crop_y = self.ws_camera_y - self.ws_crop_h // 2

    def _derive_space_vars(self):
        # world space vars
        self._derive_ws_crop()
        self.ws_canvas_x = self.ws_camera_x - self.canvas_w // 2
        self.ws_canvas_y = self.ws_camera_y - self.canvas_h // 2

        # zoom vars in world space and mask space
        self.ms_zoom_left = max(0, int(math.floor(self.ws_canvas_x / self.zoom)))
        self.ms_zoom_top = max(0, int(math.floor(self.ws_canvas_y / self.zoom)))
        self.ws_zoom_left = self.ms_zoom_left * self.zoom
        self.ws_zoom_top = self.ms_zoom_top * self.zoom
        self.ms_zoom_right = int(math.ceil((self.ws_zoom_left + self.ws_crop_w) / self.zoom))
        self.ms_zoom_bottom = int(math.ceil((self.ws_zoom_top + self.ws_crop_h) / self.zoom))
        self.ws_zoom_right = self.ms_zoom_right * self.zoom
        self.ws_zoom_bottom = self.ms_zoom_bottom * self.zoom

        # canvas space vars
        self.cs_crop_x = self.ws_crop_x - self.ws_canvas_x
        self.cs_crop_y = self.ws_crop_y - self.ws_canvas_y

    def debug_out(self):
        w = 16
        print("##############################################################")
        print()
        print("{}\t{}".format("self.zoom".ljust(w), self.zoom))
        print("{}\t{}".format("self.mask_w".ljust(w), self.mask_w))
        print("{}\t{}".format("self.mask_h".ljust(w), self.mask_h))
        print("{}\t{}".format("self.world_w".ljust(w), self.world_w))
        print("{}\t{}".format("self.world_h".ljust(w), self.world_h))
        print("{}\t{}".format("self.canvas_w".ljust(w), self.canvas_w))
        print("{}\t{}".format("self.canvas_h".ljust(w), self.canvas_h))
        print()
        print("{}\t{}".format("self.ws_camera_x".ljust(w), self.ws_camera_x))
        print("{}\t{}".format("self.ws_camera_y".ljust(w), self.ws_camera_y))
        print()
        print("{}\t{}".format("self.ws_crop_w".ljust(w), self.ws_crop_w))
        print("{}\t{}".format("self.ws_crop_h".ljust(w), self.ws_crop_h))
        print("{}\t{}".format("self.ws_crop_x".ljust(w), self.ws_crop_x))
        print("{}\t{}".format("self.ws_crop_y".ljust(w), self.ws_crop_y))
        print()
        print("{}\t{}".format("self.ws_canvas_x".ljust(w), self.ws_canvas_x))
        print("{}\t{}".format("self.ws_canvas_y".ljust(w), self.ws_canvas_y))
        print()
        print("{}\t{}".format("self.ws_zoom_left".ljust(w), self.ws_zoom_left))
        print("{}\t{}".format("self.ws_zoom_top".ljust(w), self.ws_zoom_top))
        print("{}\t{}".format("self.ws_zoom_right".ljust(w), self.ws_zoom_right))
        print("{}\t{}".format("self.ws_zoom_bottom".ljust(w), self.ws_zoom_bottom))
        print()
        print("{}\t{}".format("self.ms_zoom_left".ljust(w), self.ms_zoom_left))
        print("{}\t{}".format("self.ms_zoom_top".ljust(w), self.ms_zoom_top))
        print("{}\t{}".format("self.ms_zoom_right".ljust(w), self.ms_zoom_right))
        print("{}\t{}".format("self.ms_zoom_bottom".ljust(w), self.ms_zoom_bottom))
        print()
        print("{}\t{}".format("self.cs_crop_x".ljust(w), self.cs_crop_x))
        print("{}\t{}".format("self.cs_crop_y".ljust(w), self.cs_crop_y))
