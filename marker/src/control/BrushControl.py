from tkinter import *


class BrushControl:
    def __init__(self, master, model, painter):
        super().__init__()
        self.master = master
        self.model = model
        self.model.subject.layer.attach(self._update_layer)
        self.model.subject.project.attach(self._update_project)
        
        self.painter = painter ##

        self._reset()

    def _reset(self):
        self._next_row = 0
        self._active_indicators = []
        self._viz_boxes = []
        self._lock_boxes = []
        self._mask_opacity = None

    def _update_layer(self):
        for z in range(self.model.project.numMasks):
            layer = self.model.project.get_layer_by_z(z)

            if self.model.project.activeMask == z:
                self._active_indicators[z].config(text='Active')
            else:
                self._active_indicators[z].config(text='')

            if layer.isVisible:
                self._viz_boxes[z].select()
            else:
                self._viz_boxes[z].deselect()

            if layer.isLocked:
                self._lock_boxes[z].select()
            else:
                self._lock_boxes[z].deselect()

            if self.model.project.maskOpaque:
                self._mask_opacity.select()
            else:
                self._mask_opacity.deselect()

    def _update_project(self):
        for widget in self.master.winfo_children():
            widget.grid_forget()
            widget.destroy()
        if self.model.isProjectLoaded:
            self._reset()
            self._add_mask_opacity()
            self._add_hsi_slider()
            self._add_brush_slider()
            self._add_active_layer_controls()

            # initialize active indicators and viz boxes
            self._update_layer()

    def _add_mask_opacity(self):
        Label(self.master, text='Layer Opacity').grid(row=self._next_row, column=0)
        self._mask_opacity = Checkbutton(self.master, text='opaque', command=self.model.toggle_mask_opacity)
        self._mask_opacity.grid(row=self._next_row, column=1)
        self._next_row += 1

    def _add_brush_slider(self):
        Label(self.master, text='Brush Radius').grid(row=self._next_row, column=0)
        slider = Scale(self.master, from_=self.model.brushSizeMin, to=self.model.brushSizeMax,
                       command=self.model.set_brush_size, orient=HORIZONTAL, length=50) # self.painter.render_canvas_image    command=self.model.set_brush_size
        slider.set(self.model.brushSize)
        slider.grid(row=self._next_row, column=1, columnspan=3, ipadx=30)
        self._next_row += 1
        
    def _add_hsi_slider(self):
        Label(self.master, text='HSI Layer').grid(row=self._next_row, column=0)
        slider = Scale(self.master, from_=1, to=100, # to=self.model.hsiSizeMax
                       command=self.painter.model.set_hsi_size, orient=HORIZONTAL, length=50) #  self.model.set_hsi_size       self.painter.render_canvas_image()    command=self.model.project.hsi_MAXhsilayer, orient=HORIZONTAL, length=50         self.hsiSizeMin = 1   self.hsiSizeMax = self.project.hsi_MAXhsilayer
        slider.set(self.model.hsiSize)
        slider.grid(row=self._next_row, column=1, columnspan=3, ipadx=30)
        self._next_row += 1
    
    def _add_active_layer_controls(self):
        def grid_row(*args):
            for c in range(len(args)):
                args[c].grid(row=self._next_row, column=c)
            self._next_row += 1

        def grid_row_offset(offset, *args):
            for c in range(len(args)):
                args[c].grid(row=self._next_row, column=c + offset)
            self._next_row += 1

        label = Label(self.master, text='Active Layer')
        show_label = Label(self.master, text='show')
        lock_label = Label(self.master, text='lock')

        layer_rows = []
        for z in range(self.model.project.numMasks):
            layer = self.model.project.get_layer_by_z(z)
            layer_label = Label(self.master, width='10', text=layer.name)
            color_button = Button(self.master, width='8', bg=layer.color,
                                  command=lambda i=z: self.model.set_active_layer(i))
            layer_viz = Checkbutton(self.master,
                                    command=lambda i=z: self.model.toggle_layer_visibility(i))
            layer_lock = Checkbutton(self.master,
                                     command=lambda i=z: self.model.toggle_layer_lock(i))
            layer_rows.append((layer_label, color_button, layer_viz, layer_lock))

            self._active_indicators.append(color_button)
            self._viz_boxes.append(layer_viz)
            self._lock_boxes.append(layer_lock)

        # add the layers in the same order as PhotoShop
        # so those at the top will be in the foreground
        # and those at the bottom will be in the background
        grid_row(label)
        grid_row_offset(2, *[show_label, lock_label])
        for z in range(self.model.project.numMasks - 1, -1, -1):
            grid_row(layer_rows[z][0], layer_rows[z][1], layer_rows[z][2], layer_rows[z][3])
