from tkinter import filedialog, messagebox, simpledialog
import copy

from src.ObservableSubject import ObservableSubject
from src.model.ProjectModel import ProjectModel
from src.model.CanvasModel import CanvasModel
from src.model.KeyboardModel import KeyboardModel
from src.model.UndoHistory import UndoHistory

import os
import numpy as np
import cv2 as cv


class Subject:
    def __init__(self):
        self.load = ObservableSubject()
        self.project = ObservableSubject()
        self.layer = ObservableSubject()
        self.zoom = ObservableSubject()
        self.undo = ObservableSubject()
        self.save = ObservableSubject()
        self.export = ObservableSubject()


class Model:
    def __init__(self, master=None):
        if master:
            self.subject = Subject()
            self.project = ProjectModel()
            self.canvas = CanvasModel()
            self.keyboard = KeyboardModel(master)
            self._history = UndoHistory(20)

        # save subject
        self.isCurrentSaved = True

        # undo subject
        self._history.reset()

        # project subject
        self.isProjectLoaded = False
        self.project.unload()

        # integers
        self.brushSize = 10
        self.brushSizeMin = 2
        self.brushSizeMax = 40
        
        self.hsiSize = 1
        self.hsiSizeMin = 1 # self.project.
        self.hsiSizeMax = self.project.hsi_MAXhsilayer
        
        self.project_file_path = ""
        self.bg_image_file_path = ""
        self.key_answer = ""
        
        
        self.hsi = []
        self.bg_filename = ""
        self.backgroundImagePath = ""        
        
       
        

    def _reset(self):
        self.__init__()
        print("a")

    def has_undo(self):
        return self.isProjectLoaded and self._history.has_undo()

    def has_redo(self):
        return self.isProjectLoaded and self._history.has_redo()

    # tk scale slider events are strings that contain floats
    # convert to int before storing
    def set_brush_size(self, event):
        self.brushSize = int(float(event))
        

    def set_hsi_size(self, event):
        self.hsiSize = int(float(event))
        #print(self.hsiSize)
        #self.project.create_project(project_file_path= self.project_file_path, bg_image_file_path= self.bg_image_file_path, key_answer= self.key_answer, hsi_layer_num=self.hsiSize)
        #self.load_project(self.project_file_path)

        
        #self.load_HSI_layer(project_file_path=self.project_file_path, bg_image_file_path=self.bg_image_file_path, hsi_layer_num=self.hsiSize,key_answer=self.key_answer)
       
        '''
        bg_filename = self.project._generate_background_file_name(self.project.projectName)
        self.bg_filename = bg_filename
        self.backgroundImagePath = os.path.join(self.project.maskRootDir, bg_filename)
        
       
        config = self.project._read_default_config(self.project._config_file_path)        
        bg_img_size = self.project._copy_background_image(config, self.bg_image_file_path, self.key_answer, self.hsiSize)
        '''

        #self.load_HSI_layer(self.project_file_path, self.bg_image_file_path, self.key_answer, self.hsiSize)
        
        print(self.project_file_path, self.bg_image_file_path, self.key_answer, self.hsiSize)
        
        
        self.project.create_project(self.project_file_path, self.bg_image_file_path, self.key_answer, self.hsiSize)
        #self.load_project(self.project_file_path)

        
        #cv_bg = bg_img_size[2]
        #self.hsi = cv_bg
        
        #cv_bg = np.array(cv_bg[self.hsiSize])
        #cv.imwrite(self.backgroundImagePath, cv_bg)

        #hsi_data = self.hsi_data
        #print(np.array(hsi_data[1]))
        #cv_bg = np.array(hsi_data[1]) # self.hsiSize
        #cv.imwrite(self.backgroundImagePath, cv_bg)
        

    # change layer data and notify observers
    def toggle_mask_opacity(self):
        self.save_undo_state()
        self.project.toggle_mask_opacity()
        self._notify_needs_save()

    def set_active_layer(self, z):
        self.save_undo_state()
        self.project.set_active_layer(z)
        self._notify_needs_save()

    def toggle_layer_visibility(self, z):
        self.save_undo_state()
        self.project.toggle_layer_visibility(z)
        self._notify_needs_save()

    def toggle_layer_lock(self, z):
        self.save_undo_state()
        self.project.toggle_layer_lock(z)
        self._notify_needs_save()

    def set_layer_name(self, z, name):
        self.save_undo_state()
        self.project.get_layer_by_z(z).name = name
        self._notify_needs_save()

    def set_layer_color(self, z, color):
        self.save_undo_state()
        self.project.get_layer_by_z(z).color = color
        self._notify_needs_save()

    def set_mask_edited(self):
        self.isCurrentSaved = False
        self.subject.save.notify()

    def export_comp_image(self, pil_image):
        self.project.export_comp_image(pil_image)
        self.subject.export.notify()

    def add_layer(self, layer):
        self.save_undo_state()
        self.project.insert_layer(layer)
        self._notify_needs_save()

    def remove_layer(self, layer):
        # do not allow removing the last mask, what would happen to activeMask
        if self.project.numMasks > 1:
            self.save_undo_state()
            self.project.remove_layer(layer)
            self._notify_needs_save()

    def zoom(self, zoom_factor):
        self.canvas.set_zoom(zoom_factor) 
        self.subject.zoom.notify()

    def mouse_zoom(self, zoom_factor, e):
        self.canvas.set_mouse_zoom(zoom_factor, e)
        self.subject.zoom.notify()

    def save_undo_state(self):
        project = copy.deepcopy(self.project)
        self._history.save_state(project)
        self.subject.undo.notify()

    def _notify_needs_save(self):
        self.isCurrentSaved = False
        self.subject.project.notify()
        self.subject.layer.notify()
        self.subject.save.notify()
        self.subject.undo.notify()

    ################################
    #
    #  menu click handlers
    #
    ################################

    def undo(self):
        if self._history.has_undo():
            self.project = self._history.undo(self.project)
            self._notify_needs_save()

    def redo(self):
        if self._history.has_redo():
            self.project = self._history.redo(self.project)
            self._notify_needs_save()

    def move_active(self, before, after):
        if before != after and after >= 0 and after < self.project.numMasks:
            self.save_undo_state()
            self.project.move_active(before, after)
            self._notify_needs_save()

    def save(self):
        self.project.save()
        self.isCurrentSaved = True
        self.subject.save.notify()

    def prompt_save_as(self):
        project_file_path = filedialog.asksaveasfilename(title="Сохранить файл МАСКИ с HSI подложкой как..",
                                                         defaultextension=".mat",
                                                         filetypes=[(".mat HSI file", ".mat"),(".h5 HSI file", "*.h5"),(".npy HSI file", "*.npy")])
        if project_file_path:
            self.project.save_as(project_file_path)
            self._reset()

            self.load_project(project_file_path)

    def unload_project(self):
        if not self.isCurrentSaved:
            is_ok = messagebox.askyesno("Закрыть файл", "Вы потеряете несохранённые данные. Вы уверены?")
            if not is_ok:
                return

        self._reset()
        self.subject.load.notify()
        self.subject.project.notify()
        self.subject.save.notify()
        self.subject.undo.notify()
        stop_here=1

    def prompt_load_project(self):
        if not self.isCurrentSaved:
            is_ok = messagebox.askyesno("Открыть МАСКУ с HSI подложкой", "Вы потеряете несохранённые данные. Вы уверены?")
            if not is_ok:
                return

        proj_path = filedialog.askopenfilename()
        if proj_path:
            self.load_project(proj_path)
            
            
    def prompt_load_mask(self):
        if not self.isCurrentSaved:
            is_ok = messagebox.askyesno("Загрузка МАСКИ", "Вы потеряете несохранённую МАСКУ. Вы уверены?")
            if not is_ok:
                return

        proj_path = filedialog.askopenfilename()
        if proj_path:
            self.load_mask(proj_path) # self.load_project(proj_path)

    def load_project(self, project_file_path):
        self.project.load_project(project_file_path)
        self.canvas.resize_canvas(self.project.imgSize)

        self.isProjectLoaded = True
        self.isCurrentSaved = True
        self.subject.load.notify() # обновляем отображаемый бэкграунд
        self.subject.project.notify() # загружаем отображение ползунков слоёв и размера кисти
        self.subject.undo.notify()

    def load_mask(self, project_file_path):
        bg_filename = self.project._generate_background_file_name(self.project.projectName)
        self.bg_filename = os.path.join(self.project.maskRootDir, bg_filename)  
        
        print("!!!!!!!!  " + self.bg_filename)           #self.project_file_path = ""     self.bg_image_file_path = ""   self.key_answer = ""        
        self.project.load_mask(project_file_path, self.bg_filename)

        self.canvas.resize_canvas(self.project.imgSize)

        self.isProjectLoaded = True
        self.isCurrentSaved = True
        self.subject.load.notify()
        self.subject.project.notify()
        # self.subject.undo.notify()
        
    def load_HSI_layer(self, project_file_path, bg_image_file_path, key_answer, hsi_layer_num):
        self.project.load_HSI_layer(project_file_path, bg_image_file_path, key_answer, hsi_layer_num)
        #self.canvas.resize_canvas(self.project.imgSize)

        #self.isProjectLoaded = True
        #self.isCurrentSaved = True
        #self.subject.load.notify()
        #self.subject.project.notify()
 
        

    def prompt_create_project(self):
        if not self.isCurrentSaved:
            is_ok = messagebox.askyesno("Open Project", "You will lose any unsaved data. Are you sure?")
            if not is_ok:
                return

        project_file_path = filedialog.asksaveasfilename(title="Create a project json file",
                                                         defaultextension=".json",
                                                         filetypes=[("json files", "*.json")])
        if project_file_path:
            self.project.create_project(project_file_path)
            self.load_project(project_file_path)


    def prompt_create_background_project(self):
        if not self.isCurrentSaved:
            is_ok = messagebox.askyesno("Open Project", "You will lose any unsaved data. Are you sure?")
            if not is_ok:
                return

        project_file_path = filedialog.asksaveasfilename(title="Create a project file",
                                                         defaultextension=".json",
                                                         filetypes=[("json files", "*.json")])
        if project_file_path:
            bg_image_file_path = filedialog.askopenfilename(title="Open background HSI file",
                                                            defaultextension=".jpg",
                                                            filetypes=[("jpg files", "*.mat")]) # filetypes=[("jpg files", "*.jpg")]    .mat
            key_answer = simpledialog.askstring("Key name", "Enter key of HSI")            
        
         
            if bg_image_file_path:
                self.project.create_project(project_file_path, bg_image_file_path, key_answer)
                self.load_project(project_file_path)
                
    def prompt_create_background_and_clear_mask(self):
        if not self.isCurrentSaved:
            is_ok = messagebox.askyesno("Создать новую МАСКУ с HSI подложкой", "Вы потеряете несохранённые данные. Вы уверены?")
            if not is_ok:
                return

        project_file_path = filedialog.asksaveasfilename(title="Создать новую МАСКУ с HSI подложкой",
                                                        defaultextension=".mat",
                                                         filetypes=[(".mat files", ".mat"),(".h5 files", "*.h5"),(".npy files", "*.npy")])
        if project_file_path:
            bg_image_file_path = filedialog.askopenfilename(title="Открыть HSI подложку",
                                                            defaultextension=".mat",
                                                            filetypes=[(".mat HSI file", ".mat"),(".h5 HSI file", "*.h5"),(".npy HSI file", "*.npy")]) # filetypes=[("jpg files", "*.jpg")]    .mat
            key_answer = simpledialog.askstring("Key name", "Введите Key_of_HSI\n                                                                        ")
                    
            if bg_image_file_path:
                self.project.create_project(project_file_path, bg_image_file_path, key_answer)
                self.load_project(project_file_path)
                
                self.project_file_path = project_file_path
                self.bg_image_file_path = bg_image_file_path    
                self.key_answer = key_answer
                
                
    def prompt_create_HSI_background_only(self):
        if not self.isCurrentSaved:
            is_ok = messagebox.askyesno("Загрузить HSI подложку", "Вы потеряете несохранённые данные. Вы уверены?")
            if not is_ok:
                return

        project_file_path = filedialog.asksaveasfilename(title="Загрузить новую HSI подложку",
                                                        defaultextension=".mat",
                                                         filetypes=[(".mat files", ".mat"),(".h5 files", "*.h5"),(".npy files", "*.npy")])
        if project_file_path:
            bg_image_file_path = filedialog.askopenfilename(title="Открыть HSI подложку",
                                                            defaultextension=".mat",
                                                            filetypes=[(".mat HSI file", ".mat"),(".h5 HSI file", "*.h5"),(".npy HSI file", "*.npy")]) # filetypes=[("jpg files", "*.jpg")]    .mat
            key_answer = simpledialog.askstring("Key name", "Введите Key_of_HSI\n                                                                        ")
                    
            if bg_image_file_path:
                self.project.create_project(project_file_path, bg_image_file_path, key_answer)
                self.load_project(project_file_path)
                
                self.project_file_path = project_file_path
                self.bg_image_file_path = bg_image_file_path    
                self.key_answer = key_answer
                



                

