from tkinter import *
from tkinter import simpledialog, colorchooser, messagebox

from src.Utils import Utils


class LayerControl:
    def __init__(self, master, model):
        super().__init__()
        self.master = master
        self.model = model
        self.model.subject.project.attach(self._update_project)

        self._reset()

    def _reset(self):
        self._next_row = 0

    def _update_project(self):
        for widget in self.master.winfo_children():
            widget.grid_forget()
            widget.destroy()
        if self.model.isProjectLoaded:
            self._reset()
            self._add_layer_controls()

    def _add_layer_controls(self):
        def grid_row(*args):
            for c in range(len(args)):
                args[c].grid(row=self._next_row, column=c)
            self._next_row += 1

        def grid_row_offset(offset, *args):
            for c in range(len(args)):
                args[c].grid(row=self._next_row, column=c + offset)
            self._next_row += 1

        label = Label(self.master, text='Edit layers')

        bg_add_button = Button(self.master, width='2', text='+',
                               command=lambda layer=0: self.model.add_layer(layer))

        layer_rows = []
        
        h_eight='2'
        
        for z in range(self.model.project.numMasks):
            layer = self.model.project.get_layer_by_z(z)

            name_button = Button(self.master, width='10', height=h_eight, text=layer.name,
                                 command=lambda i=z: self._prompt_layer_name(i))
            color_button = Button(self.master, width='3', height=h_eight, bg=layer.color,
                                  command=lambda i=z: self._prompt_layer_color_chooser(i))
            hex_button = Button(self.master, width='7', height=h_eight, text=layer.color,
                                command=lambda i=z: self._prompt_layer_hex_color(i))
            add_button = Button(self.master, width='2', height=h_eight, text='+',
                                command=lambda i=z+1: self.model.add_layer(i))
            remove_button = Button(self.master, width='2', height=h_eight, text='-',
                                   command=lambda i=z: self.model.remove_layer(i))

            if self.model.project.numMasks == 1 or layer.isLocked:
                remove_button.config(state=DISABLED)

            layer_rows.append([name_button, color_button, hex_button, add_button, remove_button])

        # add the layers in the same order as PhotoShop
        # so those at the top will be in the foreground
        # and those at the bottom will be in the background
        # add_label_row(label)
        grid_row(label)
        for z in range(self.model.project.numMasks - 1, -1, -1):
            grid_row(*layer_rows[z])
        grid_row_offset(3, *[bg_add_button])

    ################################
    #
    #  click handlers
    #
    ################################

    def _prompt_layer_name(self, layer):
        answer = simpledialog.askstring("Change layer name", "Enter new name of layer")
        if answer:
            self.model.set_layer_name(layer, answer)

    def _prompt_layer_color_chooser(self, layer):
        _, answer = colorchooser.askcolor(title="Choose a new layer color")
        if answer:
            self.model.set_layer_color(layer, answer)

    def _prompt_layer_hex_color(self, layer):
        answer = simpledialog.askstring("Choose a new layer color",
                                        "Enter new hex color of layer.\nEnter 3 or 6 hexdigits.")
        if answer:
            if Utils.is_hex_color(answer):
                if answer[0] != '#':
                    answer = "#{}".format(answer)
                self.model.set_layer_color(layer, answer.lower())
            else:
                is_ok = messagebox.askyesno("Invalid entry", "{} is not a valid hex color. Try again?".format(answer))
                if is_ok:
                    self._prompt_layer_hex_color(layer)
