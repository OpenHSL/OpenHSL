from tkinter import *
from tkinter import Menu

from src.Utils import Utils


class MenuBar:
    def __init__(self, master, model, painter, app):
        self.app = app
        self.master = master
        self.model = model
        self.model.subject.project.attach(self._update_project)
        self.model.subject.layer.attach(self._update_layer)
        self.model.subject.undo.attach(self._update_undo)

        # toggle lists to control normal/disabled state of menu buttons and keyboard shortcuts
        self._disable_loaded = []
        self._disable_unloaded = []
        self._disable_no_undo = []
        self._disable_no_redo = []
        self._disable_is_top = []
        self._disable_is_bottom = []

        top_menu = Menu(self.master)
        self.master.config(menu=top_menu)

        file_menu = Menu(top_menu)
        top_menu.add_cascade(label="File", menu=file_menu)
        self._init_file_menu(file_menu, painter)

        edit_menu = Menu(top_menu)
        top_menu.add_cascade(label="Edit", menu=edit_menu)
        self._init_edit_menu(edit_menu)

        # debug_menu = Menu(top_menu)
        # top_menu.add_cascade(label="Debug", menu=debug_menu)
        # self._init_debug_menu(debug_menu)

    def _init_file_menu(self, menu, painter):
        #menu.add_command(label="New", command=self._kb_create_project, accelerator="Ctrl+N") #Ctrl+Shift+N
        menu.add_command(label="New + BG HSI", command=self._kb_create_background_project, accelerator="Ctrl+N")
        menu.add_command(label="Open", command=self._kb_load_project, accelerator="Ctrl+O")
        menu.add_command(label="Save", command=self._kb_save, accelerator="Ctrl+S")
        menu.add_command(label="Save as", command=self._kb_save_as, accelerator="Ctrl+Shift+S")
        
        menu.add_separator()
        menu.add_command(label="Close Project", command=self.model.unload_project)
        menu.add_command(label="Export Image", command=painter.export_comp_image)
        menu.add_separator()
        menu.add_command(label="Exit", command=self.master.quit)

        # add state toggles for buttons/keyboard controls depending on PROJECT update
        self._add_toggle(self._disable_unloaded, menu, menu.index("Save"), '<Control-s>', self._kb_save)
        self._add_toggle(self._disable_unloaded, menu, menu.index("Save as"), '<Control-S>', self._kb_save_as)
        self._add_toggle(self._disable_unloaded, menu, menu.index("Close Project"))
        self._add_toggle(self._disable_unloaded, menu, menu.index("Export Image"))

        # for testing that inverse works
        # self._add_toggle(self._disable_loaded, menu, menu.index("Open"), '<Control-o>', self._kb_open)

        # bind the keyboard shortcuts that will always be on
        self.master.bind('<Control-o>', self._kb_load_project)
        self.master.bind('<Control-n>', self._kb_create_project)
        self.master.bind('<Control-N>', self._kb_create_background_project)

    def _init_edit_menu(self, menu):
        menu.add_command(label="Undo", command=self._kb_undo, accelerator="Ctrl+Z", state=DISABLED)
        menu.add_command(label="Redo", command=self._kb_redo, accelerator="Ctrl+Y", state=DISABLED)
        menu.add_separator()
        menu.add_command(label="Move Active to Top", command=self._kb_move_top, accelerator="Ctrl+Shift+Up")
        menu.add_command(label="Move Active Up", command=self._kb_move_up, accelerator="Ctrl+Up")
        menu.add_command(label="Move Active Down", command=self._kb_move_down, accelerator="Ctrl+Down")
        menu.add_command(label="Move Active to Bottom", command=self._kb_move_bottom, accelerator="Ctrl+Shift+Down")

        self._add_toggle(self._disable_no_undo, menu, menu.index("Undo"), '<Control-z>', self._kb_undo)
        self._add_toggle(self._disable_no_redo, menu, menu.index("Redo"), '<Control-y>', self._kb_redo)

        self._add_toggle(self._disable_is_top, menu, menu.index("Move Active to Top"),
                         '<Control-Shift-Up>', self._kb_move_top)
        self._add_toggle(self._disable_is_top, menu, menu.index("Move Active Up"),
                         '<Control-Up>', self._kb_move_up)
        self._add_toggle(self._disable_is_bottom, menu, menu.index("Move Active Down"),
                         '<Control-Down>', self._kb_move_down)
        self._add_toggle(self._disable_is_bottom, menu, menu.index("Move Active to Bottom"),
                         '<Control-Shift-Down>', self._kb_move_bottom)

    def _init_debug_menu(self, menu):
        menu.add_command(label="Break App", command=self.app.breakpoint_app)
        menu.add_command(label="CanvasModel", command=self.model.canvas.debug_out)

    @staticmethod
    def _add_toggle(toggle_list, menu, index, key=None, on_key_press=None):
        toggle_list.append((menu, index, key, on_key_press))

    def _set_item_state(self, toggle_list, state):
        for menu, index, key, on_key_press in toggle_list:
            menu.entryconfigure(index, state=state)
            if key and on_key_press:
                if state == NORMAL:
                    self.master.bind(key, on_key_press)
                else:
                    self.master.unbind(key)

    def _update_project(self):
        if self.model.isProjectLoaded:
            self._set_item_state(self._disable_unloaded, NORMAL)
            self._set_item_state(self._disable_loaded, DISABLED)
        else:
            self._set_item_state(self._disable_unloaded, DISABLED)
            self._set_item_state(self._disable_loaded, NORMAL)
        self._update_layer()

    def _update_layer(self):
        if not self.model.isProjectLoaded:
            self._set_item_state(self._disable_is_top, DISABLED)
            self._set_item_state(self._disable_is_bottom, DISABLED)
        elif self.model.project.activeMask == 0:
            self._set_item_state(self._disable_is_top, NORMAL)
            self._set_item_state(self._disable_is_bottom, DISABLED)
        elif self.model.project.activeMask == self.model.project.numMasks - 1:
            self._set_item_state(self._disable_is_top, DISABLED)
            self._set_item_state(self._disable_is_bottom, NORMAL)
        else:
            self._set_item_state(self._disable_is_top, NORMAL)
            self._set_item_state(self._disable_is_bottom, NORMAL)

    def _update_undo(self):
        if self.model.has_undo():
            self._set_item_state(self._disable_no_undo, NORMAL)
        else:
            self._set_item_state(self._disable_no_undo, DISABLED)

        if self.model.has_redo():
            self._set_item_state(self._disable_no_redo, NORMAL)
        else:
            self._set_item_state(self._disable_no_redo, DISABLED)

    # keyboard shortcut and button handlers
    def _kb_create_background_project(self, _=None):
        self.model.prompt_create_background_project()

    def _kb_create_project(self, _=None):
        self.model.prompt_create_project()

    def _kb_load_project(self, _=None):
        self.model.prompt_load_project()

    def _kb_save(self, _=None):
        self.model.save()

    def _kb_save_as(self, _=None):
        self.model.prompt_save_as()

    def _kb_undo(self, _=None):
        self.model.undo()

    def _kb_redo(self, _=None):
        self.model.redo()

    def _kb_move_top(self, _=None):
        self.model.move_active(self.model.project.activeMask, self.model.project.numMasks - 1)

    def _kb_move_bottom(self, _=None):
        self.model.move_active(self.model.project.activeMask, 0)

    def _kb_move_up(self, _=None):
        self.model.move_active(self.model.project.activeMask, self.model.project.activeMask + 1)

    def _kb_move_down(self, _=None):
        self.model.move_active(self.model.project.activeMask, self.model.project.activeMask - 1)
