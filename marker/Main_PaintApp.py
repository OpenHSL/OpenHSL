from tkinter import *
from tkinter import Tk, Frame
from tkinter.ttk import Notebook, Style
import platform

from src.model.Model import Model
from src.view.CanvasPainter import CanvasPainter
from src.control.MenuBar import MenuBar
from src.control.StatusPanel import StatusPanel
from src.control.BrushControl import BrushControl
from src.control.LayerControl import LayerControl
from src.control.CanvasControl import CanvasControl


class TkPaintApp(Frame):
    def __init__(self, master):
        super().__init__()
        self._appTitle = "Class Painter"
        self.master = master
        self.master.title(self._appTitle)

        # vars used for widget sizes and root geometry
        canvas_width = 840    # 259
        canvas_height = 760   # 255
        
        # color of widget
        color_bg = "#dadada"
        color_dw_line = "#dadada"
        color_frame = "#eeeeee"

        # the only 2 supported systems are Windows 7 and Windows 10
        # tkinter widgets in the control panel render at different widths on Windows 7 and Windows 10
        # so change the panel width to accommodate the widgets
        if platform.system() == "Windows" and platform.release() == "7":
            self.control_panel_width = 200
        else:
            self.control_panel_width = 226

        # model must be initialized before widgets and controllers
        model = Model(master)
        model.subject.save.attach(self._update_save)
        model.subject.load.attach(self._update_load)
        model.subject.project.attach(self._update_project)
        self.model = model

        # set the widgets
        status_label = Label(self.master, text="status label...", bg=color_dw_line, bd=1, relief=SUNKEN, anchor=W)
        canvas = Canvas(self.master, width=canvas_width, height=canvas_height,
                        highlightthickness=0, bg=color_bg, relief=RAISED) # "gray50"

        
        '''
        mygreen = "#d2ffd2"
        myred = "#dd0202"
        style = Style()

        style.theme_create( "yummy", parent="alt", settings={
                "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
                "TNotebook.Tab": {
                    "configure": {"padding": [50, 5], "background": mygreen },
                    "map":       {"background": [("selected", myred)],
                                "expand": [("selected", [5, 5, 5, 0])] } } } )

        style.theme_use("yummy")
        '''

        notebook = Notebook(self.master, width=self.control_panel_width)
        brush_frame = Frame(notebook, bg=color_frame)
        layer_frame = Frame(notebook, bg=color_frame)
        notebook.add(brush_frame, text="Brush")
        notebook.add(layer_frame, text="Layers")

        # pack the widgets
        status_label.pack(side=BOTTOM, fill=X)
        notebook.pack(side=LEFT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        # canvas painter
        painter = CanvasPainter(canvas, model)

        # delegate controllers
        self.menuBar = MenuBar(self.master, model, painter, self)
        self.statusPanel = StatusPanel(status_label, model)
        self.brushControl = BrushControl(brush_frame, model, painter)
        self.layerControl = LayerControl(layer_frame, model)
        self.canvasControl = CanvasControl(canvas, model, painter)

        # set the geometry of the Tk root
        self._set_geometry(canvas_width, canvas_height)

        # to initialize the views, force a subject project notification
        model.subject.project.notify()

    def _update_save(self):
        if not self.model.isProjectLoaded:
            self.master.title(self._appTitle)
        elif self.model.isCurrentSaved:
            self.master.title("{} - {}".format(self._appTitle, self.model.project.projectFileName))
        else:
            self.master.title("{} - {} *".format(self._appTitle, self.model.project.projectFileName))

    def _update_project(self):
        if not self.model.isProjectLoaded:
            self.master.title(self._appTitle)
        else:
            self.master.title("{} - {}".format(self._appTitle, self.model.project.projectFileName))

    def _update_load(self):
        if self.model.isProjectLoaded:
            w, h = self.model.project.imgSize
            self._set_geometry(w, h)

    def _set_geometry(self, canvas_w, canvas_h):
        # set the geometry of the Tk root
        status_height = 17
        req_buffer = 4
        offset_y = -40
        width = self.control_panel_width + canvas_w + req_buffer
        height = canvas_h + status_height + req_buffer              # canvas_h + status_height + req_buffer
        x = (self.master.winfo_screenwidth() - width) // 2
        y = (self.master.winfo_screenheight() - height) // 2 + offset_y
        self.master.minsize(width, height)
        self.master.geometry("{}x{}+{}+{}".format(width, height, x, y))

    # menu bar click handlers
    def breakpoint_app(self):
        print("breakpoint app", self)


def main():

    root = Tk()
    TkPaintApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
