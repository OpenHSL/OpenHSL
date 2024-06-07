import os.path
import cv2
import numpy as np
import collections
from qimage2ndarray import rgb_view, alpha_view, array2qimage, byte_view
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QT_VERSION_STR, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QPainter, QColor, QPen
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog, QApplication

__author__ = "AI dept"
__title__ = "QTImageAnnotator"
__original_author__ = "ID25"
__original_title__ = "QtImageViewer"
__version__ = '1.6.0'

# Undo states
MAX_CTRLZ_STATES = 20

# For now, we stick to this solution.
PIXMAP_CONV_BUG_ATOL = 2

# Reusable component for painting over an image for, e.g., masking purposes
class QtImageAnnotator(QGraphicsView):

    # Mouse button signals emit image scene (x, y) coordinates.
    # !!! For image (row, column) matrix indexing, row = y and column = x.
    leftMouseButtonPressed = pyqtSignal(float, float)
    middleMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    middleMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    middleMouseButtonDoubleClicked = pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = pyqtSignal(float, float)
    mouseWheelRotated = pyqtSignal(float)

    def __init__(self):
        QGraphicsView.__init__(self)

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Shape of the loaded image (height, width)
        self.shape = (None, None)

        # Store a local handle to the scene's current image pixmap.
        self._pixmapHandle = None  # This holds the image
        self._helperHandle = None # This holds the "helper" overlay which is not directly manipulated by the user
        self._auxHelper = None  # Aux helper for various purpuses
        self._overlayHandle = []  # This is the overlay over which we are painting
        self.layer_mask = 0
        self._cursorHandle = None  # This is the cursor that appears to assist with brush size
        self._deleteCrossHandles = None # For showing that we've activated delete mode

        # Helper display state
        self.showHelper = True

        self._lastCursorCoords = None # Latest coordinates of the cursor, need in some cursor overlay update operations

        self._overlay_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        # Offscreen mask, used to speed things up (but has an impact on painting speed)
        self._offscreen_mask = None
        self._offscreen_mask_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        # Needed for proper drawing
        self.lastPoint = QPoint()
        self.lastCursorLocation = QPoint()

        # Direct mask painting
        self.direct_mask_paint = False

        # Pixmap that contains the mask and the corresponding painter
        self.mask_pixmap = None

        # Parameters of the brush and paint
        self.brush_diameter = 50
        self.MIN_BRUSH_DIAMETER = 1
        self.MAX_BRUSH_DIAMETER = 500

        self.brush_fill_color = QColor(255,0,0,99)

        # Zoom in modifier: this should be between 4 and 20
        self.zoom_in_modifier = 4

        # Painting and erasing modes
        #self.MODE_PAINT = QPainter.RasterOp_SourceOrDestination
        self.MODE_PAINT = QPainter.CompositionMode_Source
        self.MODE_ERASE = QPainter.CompositionMode_Clear
        self.current_painting_mode = self.MODE_PAINT
        self.global_erase_override = False

        # Mask related. This will allow to automatically create overlays given grayscale masks
        # and also save grayscale masks from RGB drawings. Both dicts must be provided for the
        # related functions to work properly (cannot assume unique key-value combinations)
        self.d_rgb2gray = None
        self.d_gray2rgb = None
        self.d_gray2rgb_arr = []

        # Make mouse events accessible
        self.setMouseTracking(True)

        # Image aspect ratio mode.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio

        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Stack of QRectF zoom boxes in scene coordinates.
        self.zoomStack = []

        # Flags for enabling/disabling mouse interaction.
        self.canZoom = True
        self.canPan = True
        
        self.selection = None 
        self.scenePos_x = 0
        self.scenePos_y = 0
        self.current_img = []
        self.numbox = 0
        self.min_x, self.max_x, self.min_y, self.max_y = 0, 0, 0, 0
        self.min_x_1, self.max_x_1, self.min_y_1, self.max_y_1 = 0, 0, 0, 0
        self.min_x_2, self.max_x_2, self.min_y_2, self.max_y_2 = 0, 0, 0, 0
        self.isANDImethod = False

        self.mask_pixmap_multy = []

    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    # def paintEvent(self, event):
    #     painter = QPainter(self)
    #
    #     if self._pixmapHandle is not None:
    #         painter.drawPixmap(self.rect(), self._pixmapHandle)
    #     if self._overlayHandle is not None:
    #         print("This isn't implemented yet")
    #     if self._cursorHandle is not None:
    #         painter.drawEllipse(self.rect(), self._cursorHandle)

    def clearImage(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None

    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None

    def image(self):
        """ Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None

    # Configure the annotator with data.
    # NB! Breaking change. Both IMAGE and MASK arguments from version 1.0b are
    # **assumed** to be numpy arrays!

    def clearAndSetImageOnly(self, image, helper=None, aux_helper=None,
                                process_gray2rgb=False, direct_mask_paint=False):
        # Clear the scene
        self.scene.clear()

        # Set direct mask painting mode
        self.direct_mask_paint = direct_mask_paint

        # Clear handles
        self._pixmapHandle = None
        self._helperHandle = None
        self._auxHelper = None
        self._overlayHandle = None

        # Clear UNDO stack
        self._overlay_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        # For compatibility, convert IMAGE to QImage, if needed
        if type(image) is np.array:
            image = array2qimage(image)
        
        # First we just set the image
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

        
        self.shape = pixmap.height(), pixmap.width()
        #print(self._overlay_stack, self.shape)

        self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

        # Now we add the helper, if present
        if type(helper) is np.array:
            helper = array2qimage(helper)

        if helper is not None:
            if type(helper) is QPixmap:
                pixmap = helper
            elif type(helper) is QImage:
                pixmap = QPixmap.fromImage(helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the helper layer
            self._helperHandle = self.scene.addPixmap(pixmap)

        if type(aux_helper) is np.array:
            aux_helper = array2qimage(aux_helper)

        if aux_helper is not None:
            if type(aux_helper) is QPixmap:
                pixmap = aux_helper
            elif type(aux_helper) is QImage:
                pixmap = QPixmap.fromImage(aux_helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the aux helper layer
            self._auxHelper = self.scene.addPixmap(pixmap)
            
#######################################################################################    
    def clearAndSetMaskOnly(self, image, mask, helper=None, aux_helper=None,
                                process_gray2rgb=False, direct_mask_paint=False, color=None):
        # Clear the scene
        self.scene.clear()

        # Clear handles
        self._pixmapHandle = None
        self._helperHandle = None
        self._auxHelper = None
        self._overlayHandle = None

        # Clear UNDO stack
        self._overlay_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        # For compatibility, convert IMAGE to QImage, if needed
        if type(image) is np.array:
            image = array2qimage(image)        
        # First we just set the image
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

        print(pixmap, "self.current_img !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        self.current_img = pixmap # for ai mask
        self.shape = pixmap.height(), pixmap.width()
        
        self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

        # Off-screen mask for direct drawing
        if direct_mask_paint:
            # We need to convert the offscreen mask to QImage at this point
            print(mask.data, "mask.data")
            gray_mask = QImage(mask.data, mask.shape[1], mask.shape[0], mask.strides[0], QImage.Format_Grayscale8)
            self._offscreen_mask = gray_mask.copy()
            self._offscreen_mask_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        '''
        # Now we add the helper, if present
        if type(helper) is np.array:
            helper = array2qimage(helper)

        if helper is not None:
            if type(helper) is QPixmap:
                pixmap = helper
            elif type(helper) is QImage:
                pixmap = QPixmap.fromImage(helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the helper layer
            self._helperHandle = self.scene.addPixmap(pixmap)

        if type(aux_helper) is np.array:
            aux_helper = array2qimage(aux_helper)

        if aux_helper is not None:
            if type(aux_helper) is QPixmap:
                pixmap = aux_helper
            elif type(aux_helper) is QImage:
                pixmap = QPixmap.fromImage(aux_helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the aux helper layer
            self._auxHelper = self.scene.addPixmap(pixmap)
        '''
     
        if process_gray2rgb:
            if self.d_gray2rgb:
                # We assume mask is np array, grayscale and the conversion rules are set (otherwise cannot continue)
                h, w = mask.shape
                print(h, w, mask.shape,  "h, w, mask.shape")
                
                mask = mask * list(self.d_gray2rgb.keys())[0]
                
                #import matplotlib.pyplot as plt
                #plt.plot(mask)
                #plt.show()
            
                print(list(self.d_gray2rgb.keys())[0], "list(self.d_gray2rgb.keys())[0]")
                new_mask = np.zeros((h, w, 4), np.uint8)
                for gr, rgb in self.d_gray2rgb.items():
                    print(gr, rgb, "gr, rgb ")
                    col = QColor("#63" + rgb.split("#")[1]).getRgb()# color.getRgb() # QColor("#63" + rgb.split("#")[1]).getRgb()  # TODO: not elegant, need external function
                    new_mask[mask == gr] = col
                    print(col, "col")
                
                #plt.plot(new_mask)
                #plt.show()
                
                print(new_mask, "QT - clearandsetmaskonly - if process_gray2rgb")
                
                
                use_mask = array2qimage(new_mask) #array2qimage(new_mask)
            else:
                raise RuntimeError("Cannot convert the provided grayscale mask to RGB without color specifications.")
        else:
            use_mask = array2qimage(mask)            
        
        pixmap = QPixmap.fromImage(use_mask)
        self.mask_pixmap = pixmap
        self._overlayHandle = self.scene.addPixmap(self.mask_pixmap)

        ###############################################

        # Add brush cursor to top layer
        self._cursorHandle = self.scene.addEllipse(0, 0, self.brush_diameter, self.brush_diameter)

        # Add also X to the cursor for "delete" operation, and hide it by default only showing it when the
        # either the global drawing mode is set to ERASE or when CTRL is held while drawing
        self._deleteCrossHandles = (self.scene.addLine(0, 0, self.brush_diameter, self.brush_diameter),
                                    self.scene.addLine(0, self.brush_diameter, self.brush_diameter, 0))

        if self.current_painting_mode is not self.MODE_ERASE:
            self._deleteCrossHandles[0].hide()
            self._deleteCrossHandles[1].hide()

        self.updateViewer()

#######################################################################################
    # накладывает несколько цветов по слоям маски
    def clearAndSetMaskLayers(self, image, mask, helper=None, aux_helper=None,
                                process_gray2rgb=False, 
                                direct_mask_paint=False, 
                                color=None, 
                                is_not_cube_mask = False, 
                                data_shape = True,
                                is_hidden = False):
        # Clear the scene
        self.scene.clear()

        # Clear handles
        self._pixmapHandle = None
        self._helperHandle = None
        self._auxHelper = None
        self._overlayHandle = []

        # Clear UNDO stack
        self._overlay_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        # For compatibility, convert IMAGE to QImage, if needed
        if type(image) is np.array:
            image = array2qimage(image)        
        # First we just set the image
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

        print(pixmap, "self.current_img !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.current_img = pixmap # for ai mask

        self.shape = pixmap.height(), pixmap.width()

        self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

        # Off-screen mask for direct drawing
        if direct_mask_paint:
            # We need to convert the offscreen mask to QImage at this point
            if data_shape == True:
                gray_mask = QImage(mask.data[:,:,0], mask.data.shape[1], mask.data.shape[0], mask.data.strides[0], QImage.Format_Grayscale8)
    
            else:
                gray_mask = QImage(mask[:,:,0], mask.shape[1], mask.shape[0], mask.strides[0], QImage.Format_Grayscale8)

            self._offscreen_mask = gray_mask.copy()
            self._offscreen_mask_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        '''
        # Now we add the helper, if present
        if type(helper) is np.array:
            helper = array2qimage(helper)

        if helper is not None:
            if type(helper) is QPixmap:
                pixmap = helper
            elif type(helper) is QImage:
                pixmap = QPixmap.fromImage(helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the helper layer
            self._helperHandle = self.scene.addPixmap(pixmap)

        if type(aux_helper) is np.array:
            aux_helper = array2qimage(aux_helper)

        if aux_helper is not None:
            if type(aux_helper) is QPixmap:
                pixmap = aux_helper
            elif type(aux_helper) is QImage:
                pixmap = QPixmap.fromImage(aux_helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the aux helper layer
            self._auxHelper = self.scene.addPixmap(pixmap)
            
        '''
        if is_not_cube_mask == True:
            print("sss")
            #self.scene.clear()
            '''
            print(mask, "mask")
            a1,a2 = mask.shape
            print(a1,a2, "a1,a2 FDEL ") 

            if process_gray2rgb:
                print(self.d_gray2rgb_arr, "self.d_gray2rgb")
                if self.d_gray2rgb_arr:
                    # We assume mask is np array, grayscale and the conversion rules are set (otherwise cannot continue)
                    h, w, = a1,a2 # h, w, p = mask.data.shape
                    mask_ = mask[:,:]
                    
                    print(mask_, " mask_ ")
                    
                    print(self.d_gray2rgb_arr, "self.d_gray2rgb_arr")

                    print(self.d_gray2rgb_arr.keys(), "self.d_gray2rgb_arr.keys")
                    mask_ = mask_ * list(self.d_gray2rgb_arr.keys())
                    
                    new_mask = np.zeros((h, w, 4), np.uint8)
                    
                    print(self.d_gray2rgb_arr.items(), " self.d_gray2rgb_arr.items() ")
                    for gr, rgb in self.d_gray2rgb_arr.items(): # color self.d_gray2rgb.items():
                        print (gr, rgb)
                        col = QColor("#63" + rgb.split("#")[1]).getRgb() #col = QColor("#63" + "#ddffe7".split("#")[1]).getRgb() #color.getRgb() # QColor("#63" + rgb.split("#")[1]).getRgb() 
                        new_mask[mask_ == gr] = col
                    #print(new_mask, "QT - clearandsetmasklayers - new_mask")
                    
                    print(new_mask, "new_mask")
                    #from matplotlib import pyplot as PLT
                    #PLT.imshow(new_mask, interpolation='nearest')
                    #PLT.show() 
                    
                    use_mask = array2qimage(new_mask) #array2qimage(new_mask)
                else:
                    raise RuntimeError("Cannot convert the provided grayscale mask to RGB without color specifications.")
            else:
                mask_ = mask[:,:]
                use_mask = array2qimage(mask_)            
            
            #from matplotlib import pyplot as PLT
            #PLT.imshow(mask_, interpolation='nearest')
            #PLT.show()    
            
            pixmap = QPixmap.fromImage(use_mask)
            self.mask_pixmap = pixmap
            '''
            
            self._overlayHandle.clear()
            self.mask_pixmap = QPixmap(pixmap.rect().width(), pixmap.rect().height())
            self.mask_pixmap.fill(QColor(0,0,0,0))
            self._overlayHandle = self.scene.addPixmap(self.mask_pixmap)
        
        else:
            
            if data_shape == True:
                a1,a2,num_layers = mask.data.shape
            else:
                a1,a2,num_layers = mask.shape    
                
            print(a1,a2,num_layers, "a1,a2,num_layers ") 
            for ind_layer in range(num_layers):
                if process_gray2rgb:
                    print(self.d_gray2rgb_arr, "self.d_gray2rgb")
                    if self.d_gray2rgb_arr:
                        # We assume mask is np array, grayscale and the conversion rules are set (otherwise cannot continue)
                        h, w, = a1,a2 # h, w, p = mask.data.shape
                        
                        if data_shape == True:
                            if is_hidden == True:
                                mask_ = mask[:,:,ind_layer] # mask.data[:,:,ind_layer]
                            else:
                                mask_ = mask.data[:,:,ind_layer]
                        else:
                            mask_ = mask[:,:,ind_layer]   
                

                        
                        print(mask_, " mask_ ")
                        
                        print(self.d_gray2rgb_arr, "self.d_gray2rgb_arr")

                        print(self.d_gray2rgb_arr.keys(), "self.d_gray2rgb_arr.keys")
                        mask_ = mask_ * list(self.d_gray2rgb_arr.keys())[ind_layer]
                        
                        new_mask = np.zeros((h, w, 4), np.uint8)
                        
                        print(self.d_gray2rgb_arr.items(), " self.d_gray2rgb_arr.items() ")
                        for gr, rgb in self.d_gray2rgb_arr.items(): # color self.d_gray2rgb.items():
                            print (gr, rgb)
                            col = QColor("#63" + rgb.split("#")[1]).getRgb() #col = QColor("#63" + "#ddffe7".split("#")[1]).getRgb() #color.getRgb() # QColor("#63" + rgb.split("#")[1]).getRgb() 
                            new_mask[mask_ == gr] = col
                        #print(new_mask, "QT - clearandsetmasklayers - new_mask")
                        
                        print(new_mask, "new_mask")
                        #from matplotlib import pyplot as PLT
                        #PLT.imshow(new_mask, interpolation='nearest')
                        #PLT.show() 
                        
                        use_mask = array2qimage(new_mask) #array2qimage(new_mask)
                    else:
                        raise RuntimeError("Cannot convert the provided grayscale mask to RGB without color specifications.")
                else:
                    if data_shape == True:  
                        mask_ = mask.data[:,:,ind_layer]
                    else:
                        mask_ = mask[:,:,ind_layer]
                        
                        import matplotlib.pyplot as plt
                        plt.imshow(mask_)
                        plt.show()   
                            
                    #mask_ = mask.data[:,:,ind_layer]
                    use_mask = array2qimage(mask_)            
                
                #from matplotlib import pyplot as PLT
                #PLT.imshow(mask_, interpolation='nearest')
                #PLT.show()    
                
                pixmap = QPixmap.fromImage(use_mask)
                self.mask_pixmap_multy.append(pixmap) # self.mask_pixmap_multy[ind_layer] = pixmap 
                print(self.mask_pixmap_multy[ind_layer] , "self.mask_pixmap  in MASKLAYERS")
                self._overlayHandle.append(self.scene.addPixmap(self.mask_pixmap_multy[ind_layer])) # self._overlayHandle.append(self.scene.addPixmap(self.mask_pixmap)) 
                #self._overlayHandle = self.scene.addPixmap(self.mask_pixmap)
            ###############################################

        # Add brush cursor to top layer
        self._cursorHandle = self.scene.addEllipse(0, 0, self.brush_diameter, self.brush_diameter)

        # Add also X to the cursor for "delete" operation, and hide it by default only showing it when the
        # either the global drawing mode is set to ERASE or when CTRL is held while drawing
        self._deleteCrossHandles = (self.scene.addLine(0, 0, self.brush_diameter, self.brush_diameter),
                                    self.scene.addLine(0, self.brush_diameter, self.brush_diameter, 0))

        
        if self.current_painting_mode is not self.MODE_ERASE:
            self._deleteCrossHandles[0].hide()
            self._deleteCrossHandles[1].hide()
        
        self.updateViewer()
                
 #######################################################################################           
    def clearAndSetImageAndMask(self, image, mask, helper=None, aux_helper=None,
                                process_gray2rgb=False, direct_mask_paint=False):
        # Clear the scene
        self.scene.clear()

        # Set direct mask painting mode
        self.direct_mask_paint = direct_mask_paint

        # Clear handles
        self._pixmapHandle = None
        self._helperHandle = None
        self._auxHelper = None
        self._overlayHandle = None

        # Clear UNDO stack
        self._overlay_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        # For compatibility, convert IMAGE to QImage, if needed
        if type(image) is np.array:
            image = array2qimage(image)
            print("1")        
        # First we just set the image
        if type(image) is QPixmap:
            pixmap = image
            print("2")
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
            print("3")
        else:
            raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

        self.current_img = pixmap # for ai mask
        
        self.shape = pixmap.height(), pixmap.width()
        #print(self._overlay_stack, self.shape)

        self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

        # Off-screen mask for direct drawing
        if direct_mask_paint:
            # We need to convert the offscreen mask to QImage at this point
            gray_mask = QImage(mask.data, mask.shape[1], mask.shape[0], mask.strides[0], QImage.Format_Grayscale8)
            self._offscreen_mask = gray_mask.copy()
            self._offscreen_mask_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        # Now we add the helper, if present
        if type(helper) is np.array:
            helper = array2qimage(helper)

        if helper is not None:
            if type(helper) is QPixmap:
                pixmap = helper
            elif type(helper) is QImage:
                pixmap = QPixmap.fromImage(helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the helper layer
            self._helperHandle = self.scene.addPixmap(pixmap)

        if type(aux_helper) is np.array:
            aux_helper = array2qimage(aux_helper)

        if aux_helper is not None:
            if type(aux_helper) is QPixmap:
                pixmap = aux_helper
            elif type(aux_helper) is QImage:
                pixmap = QPixmap.fromImage(aux_helper)
            else:
                raise RuntimeError("QtImageAnnotator.clearAndSetImageAndMask: Argument must be a QImage or QPixmap.")

            # Add the aux helper layer
            self._auxHelper = self.scene.addPixmap(pixmap)

        #########
        ### почему убрала ???
        # If we are supplied a grayscale mask that we need to convert to RGB, we will do it here
        
        if process_gray2rgb:
            if self.d_gray2rgb:
                # We assume mask is np array, grayscale and the conversion rules are set (otherwise cannot continue)
                h, w = mask.shape
                new_mask = np.zeros((h, w, 4), np.uint8)
                for gr, rgb in self.d_gray2rgb.items():
                    col = QColor("#63" + rgb.split("#")[1]).getRgb()  # TODO: not elegant, need external function
                    new_mask[mask == gr] = col
                use_mask = array2qimage(new_mask)
            else:
                raise RuntimeError("Cannot convert the provided grayscale mask to RGB without color specifications.")
        else:
            use_mask = array2qimage(mask)            
        
        pixmap = QPixmap.fromImage(use_mask)
        self.mask_pixmap = pixmap
        self._overlayHandle = self.scene.addPixmap(self.mask_pixmap)
        
        #########
        ### почему убрала ???

        
        ###############################################
        ###############################################        
        # Add the mask layer
        self.mask_pixmap = QPixmap(pixmap.rect().width(), pixmap.rect().height())
        self.mask_pixmap.fill(QColor(0,0,0,0))
        self._overlayHandle = self.scene.addPixmap(self.mask_pixmap)
        ###############################################
        ###############################################

        # Add brush cursor to top layer
        self._cursorHandle = self.scene.addEllipse(0, 0, self.brush_diameter, self.brush_diameter)

        # Add also X to the cursor for "delete" operation, and hide it by default only showing it when the
        # either the global drawing mode is set to ERASE or when CTRL is held while drawing
        self._deleteCrossHandles = (self.scene.addLine(0, 0, self.brush_diameter, self.brush_diameter),
                                    self.scene.addLine(0, self.brush_diameter, self.brush_diameter, 0))

        if self.current_painting_mode is not self.MODE_ERASE:
            self._deleteCrossHandles[0].hide()
            self._deleteCrossHandles[1].hide()

        self.updateViewer()

#######################################################################################
    # Clear everything
    def clearAll(self):

        self.shape = (None, None)

        if self._pixmapHandle is not None:
            self.scene.removeItem(self._pixmapHandle)

        if self._helperHandle is not None:
            self.scene.removeItem(self._helperHandle)

        if self._auxHelper is not None:
            self.scene.removeItem(self._auxHelper)

        if self._overlayHandle[self.layer_mask] is not None:
            self.scene.removeItem(self._overlayHandle[self.layer_mask])

        self._pixmapHandle = None
        self._helperHandle = None
        self._auxHelper = None
        self._overlayHandle.clear() #self._overlayHandle = []

        if self.direct_mask_paint:
            self._offscreen_mask = None
            self._offscreen_mask_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)

        self._overlay_stack = collections.deque(maxlen=MAX_CTRLZ_STATES)
        self.updateViewer()

#######################################################################################
    # Set image only
    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap | numpy.array
        """
        if type(image) is np.array:
            image = array2qimage(image)

        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)

        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.

        # Add the mask layer
        self.mask_pixmap = QPixmap(pixmap.rect().width(), pixmap.rect().height())
        self.mask_pixmap.fill(QColor(0,0,0,0))
        self._overlayHandle = self.scene.addPixmap(self.mask_pixmap)

        # Add brush cursor to top layer
        self._cursorHandle = self.scene.addEllipse(0,0,self.brush_diameter,self.brush_diameter)

        # Add also X to the cursor for "delete" operation, and hide it by default only showing it when the
        # either the global drawing mode is set to ERASE or when CTRL is held while drawing
        self._deleteCrossHandles = (self.scene.addLine(0, 0, self.brush_diameter, self.brush_diameter),
                                    self.scene.addLine(0, self.brush_diameter, self.brush_diameter, 0))

        if self.current_painting_mode is not self.MODE_ERASE:
            self._deleteCrossHandles[0].hide()
            self._deleteCrossHandles[1].hide()

        self.updateViewer()

#######################################################################################
    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
            self.fitInView(self.zoomStack[-1], self.aspectRatioMode)   # Show zoomed rect
        else:
            self.zoomStack = []  # Clear the zoom stack (in case we got here because of an invalid zoom).
            self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).
#######################################################################################
    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.updateViewer()
        
#######################################################################################
    def update_brush_diameter(self, change):
        val = self.brush_diameter
        val += change
        if val > self.MAX_BRUSH_DIAMETER:
            val = self.MAX_BRUSH_DIAMETER

        if val < self.MIN_BRUSH_DIAMETER:
            val = self.MIN_BRUSH_DIAMETER

        self.brush_diameter = val

        if self._lastCursorCoords is not None:
            x, y = self._lastCursorCoords
        else:
            x, y = 0, 0

        if self._cursorHandle is not None:
            self._cursorHandle.setPos(x - self.brush_diameter / 2, y - self.brush_diameter / 2)
            self._cursorHandle.setRect(0, 0, self.brush_diameter, self.brush_diameter)

        if self._deleteCrossHandles is not None:
            self._deleteCrossHandles[0].setLine(x - self.brush_diameter / (2 * np.sqrt(2)),
                                                y - self.brush_diameter / (2 * np.sqrt(2)),
                                                x + self.brush_diameter / (2 * np.sqrt(2)),
                                                y + self.brush_diameter / (2 * np.sqrt(2)))
            self._deleteCrossHandles[1].setLine(x - self.brush_diameter / (2 * np.sqrt(2)),
                                                y + self.brush_diameter / (2 * np.sqrt(2)),
                                                x + self.brush_diameter / (2 * np.sqrt(2)),
                                                y - self.brush_diameter / (2 * np.sqrt(2)))

#######################################################################################
    def update_cursor_location(self, event):

        scenePos = self.mapToScene(event.pos())
        x, y = scenePos.x(), scenePos.y()
        self.scenePos_x = int(scenePos.x())
        self.scenePos_y = int(scenePos.y())
        
        if event.buttons() == Qt.RightButton: 
            print("1111111111111111111111111111111111111111111111111111111111")
            self.selection.append(event.pos()) ##################### event.button() == Qt.RightButton:
        else: self.selection = [event.pos()]
        self.calculate_average_value()         

        # Store the coordinates for other operations to use
        self._lastCursorCoords = (x,y)

        if self._cursorHandle is not None:
            self._cursorHandle.setPos(x - self.brush_diameter/2, y - self.brush_diameter/2)

        if self._deleteCrossHandles is not None:
            self._deleteCrossHandles[0].setLine(x - self.brush_diameter / (2 * np.sqrt(2)),
                                                y - self.brush_diameter / (2 * np.sqrt(2)),
                                                x + self.brush_diameter / (2 * np.sqrt(2)),
                                                y + self.brush_diameter / (2 * np.sqrt(2)))
            self._deleteCrossHandles[1].setLine(x - self.brush_diameter / (2 * np.sqrt(2)),
                                                y + self.brush_diameter / (2 * np.sqrt(2)),
                                                x + self.brush_diameter / (2 * np.sqrt(2)),
                                                y - self.brush_diameter / (2 * np.sqrt(2)))

#######################################################################################
    def calculate_average_value(self):
        if self.selection is not None:
            x_values = [point.x() for point in self.selection]
            y_values = [point.y() for point in self.selection]
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)

            #print(min_x,max_x,min_y, max_y)
            self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y
            
            image = self.current_img
            width,height = self.shape
            #print(width,height, "width,height")
            #print(image, "image dddd")
            #width = image.width()
            #height = image.height()
            channels_count = 4
            image = image.toImage() # image[0].toImage()
            s = image.bits().asstring(width * height * channels_count)
            array = np.fromstring(s, dtype=np.uint8).reshape((height, width, channels_count))
            
            #print(array)
            
            #cropped_mask = self.current_img[min_x:max_x, min_y:max_y] 
            #self.selected_array = np.array(cropped_mask.toImage().constBits())
            
            #pixels = cropped_mask.toImage().constBits()
            #num_pixels = cropped_mask.width() * cropped_mask.height()
            #total_value = sum(pixels) / num_pixels

            
#######################################################################################
    def redraw_cursor(self):
        if self._cursorHandle is not None:
            self._cursorHandle.update()

        if self._deleteCrossHandles is not None:
            self._deleteCrossHandles[0].update()
            self._deleteCrossHandles[1].update()

#######################################################################################
    # Draws a single ellipse
    def fillMarker(self, event):
        scenePos = self.mapToScene(event.pos())
        painter = QPainter(self.mask_pixmap)
        painter.setCompositionMode(self.current_painting_mode)
        painter.setPen(self.brush_fill_color)
        painter.setBrush(self.brush_fill_color)

        # Get the coordinates of where to draw
        a0 = scenePos.x() - self.brush_diameter/2
        b0 = scenePos.y() - self.brush_diameter/2
        r0 = self.brush_diameter

        # Finally, draw
        painter.drawEllipse(int(a0), int(b0), r0, r0)

        # TODO: With really large images, update is rather slow. Must somehow fix this.
        # It seems that the way to approach hardcore optimization is to switch to OpenGL
        # for all rendering purposes. This update will likely come much later in the tool's
        # lifecycle.
        print(self.layer_mask, "self.layer_mask")
        self._overlayHandle[self.layer_mask].setPixmap(self.mask_pixmap)

        # In case of direct mask paint mode, we need to paint on the mask as well
        if self.direct_mask_paint:
            if not self.d_rgb2gray:
                raise RuntimeError("Cannot use direct mask painting since there is no color conversion rules set.")
            painter = QPainter(self._offscreen_mask)
            painter.setCompositionMode(self.current_painting_mode)
            tc = self.d_rgb2gray[self.brush_fill_color.name()]
            painter.setPen(QColor(tc,tc,tc))
            painter.setBrush(QColor(tc,tc,tc))
            painter.drawEllipse(a0, b0, r0, r0)

        self.lastPoint = scenePos

    # Draws a line
    def drawMarkerLine(self, event):
        scenePos = self.mapToScene(event.pos())
        print(self.mask_pixmap_multy[self.layer_mask], "self.mask_pixmap drawMarkerLine")
        
        #sel = self._overlayHandle[self.layer_mask]
        painter = QPainter(self.mask_pixmap_multy[self.layer_mask]) #
        painter.setCompositionMode(self.current_painting_mode)
        painter.setPen(QPen(self.brush_fill_color,
                            self.brush_diameter, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.lastPoint, scenePos)
        
        
        ###################### self._overlayHandle = self.scene.addPixmap(self.mask_pixmap)
        self._overlayHandle[self.layer_mask].setPixmap(self.mask_pixmap_multy[self.layer_mask])
        #print(self._overlayHandle[self.layer_mask].setPixmap(self.mask_pixmap))

        
        # In case of direct mask paint mode, we need to paint on the mask as well
        if self.direct_mask_paint:
            if not self.d_rgb2gray:
                raise RuntimeError("Cannot use direct mask painting since there is no color conversion rules set.")
            painter = QPainter(self._offscreen_mask)
            painter.setCompositionMode(self.current_painting_mode)
            tc = self.d_rgb2gray[self.brush_fill_color.name()]
            painter.setPen(QPen(QColor(tc, tc, tc),
                           self.brush_diameter, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, scenePos)

        self.lastPoint = scenePos
        

    # Fills an area using the last stored cursor location
    # If optional argument remove_closed_contour is set to True, then
    # the closed contour over which the cursor is hovering will be erased
    def fillArea(self, remove_closed_contour=False, remove_only_current_color=True):

        # Store previous state so we can go back to it
        self._overlay_stack.append(self.mask_pixmap.copy())

        if self.direct_mask_paint:
            self._offscreen_mask_stack.append(self._offscreen_mask.copy())

        # We first convert the mask to a QImage and then to ndarray
        orig_mask = self.mask_pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        msk = alpha_view(orig_mask).copy()

        # Apply simple tresholding and invert the image
        msk[np.where((msk>0))] = 255
        msk = 255-msk
        msk1 = np.copy(msk)

        if remove_closed_contour:
            msk1 = 255-msk1

        if remove_closed_contour:
            if remove_only_current_color:
                the_mask = np.ones(msk1.shape[:2], np.uint8) * 255  # Initial mask
                fullmask = self.export_ndarray_noalpha()  # Get the colored version
                reds, greens, blues = fullmask[:, :, 0], fullmask[:, :, 1], fullmask[:, :, 2]
                cur_col = list(self.brush_fill_color.getRgb())[:-1]  # Only current color is considered
                # So that fill happens only for this specific color
                the_mask[np.isclose(reds, cur_col[0], atol=PIXMAP_CONV_BUG_ATOL) &
                         np.isclose(greens, cur_col[1], atol=PIXMAP_CONV_BUG_ATOL) &
                         np.isclose(blues, cur_col[2], atol=PIXMAP_CONV_BUG_ATOL)] = 0

            else:
                the_mask = np.zeros(msk1.shape[:2], np.uint8)
        else:
            the_mask = cv2.bitwise_not(np.copy(msk))

        the_mask = cv2.copyMakeBorder(the_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

        # Fill the contour
        seed_point = (int(self.lastCursorLocation.x()), int(self.lastCursorLocation.y()))
        cv2.floodFill(msk1, the_mask, seed_point, 0, 0, 1)

        # We paint in only the newly arrived pixels (or remove the pixels in the contour)
        if remove_closed_contour:
            paintin = msk1
        else:
            paintin = msk - msk1  # This is fill case

        # Take original pixmap image: it has two components, RGB and ALPHA
        new_img = np.dstack((rgb_view(orig_mask), alpha_view(orig_mask)))

        # Fill the newly created area with current brush color
        if not remove_closed_contour:
            new_img[np.where((paintin==255))] = list(self.brush_fill_color.getRgb())
        else:
            new_img[np.where((paintin==0))] = (0,0,0,0)  # Erase
        new_qimg = array2qimage(new_img)

        # In case of direct drawing, need to update the offscreen mask as well
        if self.direct_mask_paint:
            omask = byte_view(self._offscreen_mask).copy()
            omask = omask.reshape(omask.shape[:-1])
            if not remove_closed_contour:
                tc = self.d_rgb2gray[self.brush_fill_color.name()]
                omask[np.where((paintin==255))] = tc
            else:
                omask[np.where((paintin==0))] = 0
            self._offscreen_mask = QImage(omask.data, omask.shape[1], omask.shape[0], omask.strides[0],
                                          QImage.Format_Grayscale8)

        # Finally update the screen stuff
        self.mask_pixmap = QPixmap.fromImage(new_qimg)
        self._overlayHandle[self.layer_mask].setPixmap(self.mask_pixmap)


    # Repaint connected contour (disregarding color information) to the current paint color
    def repaintArea(self):

        self._overlay_stack.append(self.mask_pixmap.copy())
        if self.direct_mask_paint:
            self._offscreen_mask_stack.append(self._offscreen_mask.copy())
        orig_mask = self.mask_pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        msk = alpha_view(orig_mask).copy()
        msk[np.where((msk>0))] = 255
        msk = 255-msk
        msk1 = 255-np.copy(msk)
        the_mask = cv2.copyMakeBorder(np.zeros(msk1.shape[:2], np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        seed_point = (int(self.lastCursorLocation.x()), int(self.lastCursorLocation.y()))
        cv2.floodFill(msk1, the_mask, seed_point, 0, 0, 1)
        paintin = np.bitwise_xor(msk, msk1)
        new_img = np.dstack((rgb_view(orig_mask), alpha_view(orig_mask)))
        new_img[np.where((paintin == 0))] = list(self.brush_fill_color.getRgb())
        new_qimg = array2qimage(new_img)

        if self.direct_mask_paint:
            omask = byte_view(self._offscreen_mask).copy()
            omask = omask.reshape(omask.shape[:-1])
            tc = self.d_rgb2gray[self.brush_fill_color.name()]
            omask[np.where((paintin == 0))] = tc
            self._offscreen_mask = QImage(omask.data, omask.shape[1], omask.shape[0], omask.strides[0],
                                          QImage.Format_Grayscale8)

        self.mask_pixmap = QPixmap.fromImage(new_qimg)
        self._overlayHandle[self.layer_mask].setPixmap(self.mask_pixmap)


    '''
    ***********************
    IMPORTERS AND EXPORTERS
    ***********************
    '''

    # Export the grayscale mask
    # This should always be used with direct mode, which supports up to 255 colors for the mask
    def export_rgb2gray_mask(self):
        if self._overlayHandle is not None:
            if self.d_rgb2gray:

                if self.direct_mask_paint:
                    # Easy mode
                    mask = byte_view(self._offscreen_mask).copy()
                    mask = mask.reshape(mask.shape[:-1])
                else:
                    # The hard way
                    # Split the image to rgb components
                    rgb_m = self.export_ndarray_noalpha()
                    reds, greens, blues = rgb_m[:, :, 0], rgb_m[:, :, 1], rgb_m[:, :, 2]
                    h, w, _ = rgb_m.shape
                    mask = np.zeros((h, w), np.uint8)

                    # Go through all the colors and paint the grayscale mask according to the conversion spec
                    for rgb, gr in self.d_rgb2gray.items():
                        cc = list(QColor(rgb).getRgb())
                        mask[np.isclose(reds, cc[0], atol=PIXMAP_CONV_BUG_ATOL) &
                             np.isclose(greens, cc[1], atol=PIXMAP_CONV_BUG_ATOL) &
                             np.isclose(blues, cc[2], atol=PIXMAP_CONV_BUG_ATOL)] = gr
            else:
                raise RuntimeError("Cannot convert the RGB mask to grayscale without color specifications.")
        else:
            raise RuntimeError("There is no RGB mask to export to grayscale.")
        return mask

    # Export current mask WITHOUT alpha channel (mask types are determined by colors, not by alpha anyway)
    def export_ndarray_noalpha(self):
        mask = self.mask_pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        return rgb_view(mask).copy()

    def export_ndarray(self):
        mask = self.mask_pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        return np.dstack((rgb_view(mask).copy(), alpha_view(mask).copy()))

    '''
    **************
    EVENT HANDLERS
    **************
    '''

    def wheelEvent(self, event):

        if self.hasImage():

            self.redraw_cursor()

            # Depending on whether control is pressed, set brush diameter accordingly
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                change = 1 if event.angleDelta().y() > 0 else -1
                self.update_brush_diameter(change)
                self.redraw_cursor()
                self.mouseWheelRotated.emit(change)
            else:
                QGraphicsView.wheelEvent(self, event)

    def mouseMoveEvent(self, event):
        ###############################################################################################

        if self.hasImage():

            # Make sure that the element has focus when the mouse moves,
            # otherwise keyboard shortcuts will not work
            if not self.hasFocus():
                self.setFocus()

            self.update_cursor_location(event)

            # Support for panning
            if event.buttons() == Qt.MiddleButton:
                offset = self.__prevMousePos - event.pos()
                self.__prevMousePos = event.pos()
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() + offset.y())
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + offset.x())

            # Filling in the markers
            if event.buttons() == Qt.LeftButton:
                self.drawMarkerLine(event)

            # Store cursor location separately; needed for certain operations (like fill)
            self.lastCursorLocation = self.mapToScene(event.pos())
            
            #if event.buttons() == Qt.RightButton:
                #print(self.scenePos_x, self.scenePos_y, "self.scenePos_x, self.scenePos_y")

        QGraphicsView.mouseMoveEvent(self, event)

    # Keypress event handler
    def keyPressEvent(self, event):

        if self.hasImage():

            # Zoom in
            if event.key() == Qt.Key_Plus:

                viewBBox = self.zoomStack[-1] if len(self.zoomStack) else self.sceneRect()

                wh12 = int(max(viewBBox.width(), viewBBox.height()) / self.zoom_in_modifier)
                x, y = self._lastCursorCoords

                
                selectionBBox = QRectF(x-wh12, y-wh12, 2*wh12, 2*wh12).intersected(viewBBox)

                if selectionBBox.isValid() and (selectionBBox != viewBBox):
                    self.zoomStack.append(selectionBBox)
                    self.updateViewer()
                

            # Zoom out
            if event.key() == Qt.Key_Minus:
                if self.canZoom:
                    viewBBox = self.zoomStack[-1] if len(self.zoomStack) else False
                    if viewBBox:
                        self.zoomStack = self.zoomStack[:-1]
                        self.updateViewer()

            # Fill mask region
            if event.key() == Qt.Key_F:
                try:
                    self.viewport().setCursor(Qt.BusyCursor)
                    self.fillArea()
                except Exception as e:
                    print("Cannot fill region. Additional information:")
                self.viewport().setCursor(Qt.ArrowCursor)

            # Erase closed contour under cursor with current paint color
            if event.key() == Qt.Key_X:
                if QApplication.keyboardModifiers() & Qt.ControlModifier:
                    try:
                        self.viewport().setCursor(Qt.BusyCursor)
                        self.fillArea(remove_closed_contour=True)
                    except Exception as e:
                        print("Cannot remove the contour. Additional information:")
                    self.viewport().setCursor(Qt.ArrowCursor)

            # Erase closed contour under cursor and any connected contour regardless of color
            if event.key() == Qt.Key_Q:
                if QApplication.keyboardModifiers() & Qt.ControlModifier:
                    try:
                        self.viewport().setCursor(Qt.BusyCursor)
                        self.fillArea(remove_closed_contour=True, remove_only_current_color=False)
                    except Exception as e:
                        print("Cannot remove the contour. Additional information:")
                    self.viewport().setCursor(Qt.ArrowCursor)

            # Erase mode enable/disable
            if event.key() == Qt.Key_D:
                self.global_erase_override = not self.global_erase_override
                if self.global_erase_override:
                    self.current_painting_mode = self.MODE_ERASE
                    self._deleteCrossHandles[0].show()
                    self._deleteCrossHandles[1].show()
                else:
                    self.current_painting_mode = self.MODE_PAINT
                    self._deleteCrossHandles[0].hide()
                    self._deleteCrossHandles[1].hide()

            # Temporarily hide the overlay
            if event.key() == Qt.Key_H:
                self._overlayHandle.hide()

            # Toggle helper on and off
            if event.key() == Qt.Key_T:
                if self._auxHelper is not None:
                    if self.showHelper:
                        self._auxHelper.hide()
                        self.showHelper = False
                    else:
                        self._auxHelper.show()
                        self.showHelper = True

            # Undo operations
            
            if event.key() == Qt.Key_Z:
                if QApplication.keyboardModifiers() & Qt.ControlModifier:
                    if (len(self._overlay_stack) > 0):
                        self.mask_pixmap = self._overlay_stack.pop()
                        self._overlayHandle[self.layer_mask].setPixmap(self.mask_pixmap)

                    if self.direct_mask_paint:
                        if len(self._offscreen_mask_stack) > 0:
                            self._offscreen_mask = self._offscreen_mask_stack.pop()

            # When CONTROL is pressed, show the delete cross
            if event.key() == Qt.Key_Control and not self.global_erase_override:
                self._deleteCrossHandles[0].show()
                self._deleteCrossHandles[1].show()

        QGraphicsView.keyPressEvent(self, event)

    def keyReleaseEvent(self, event):

        if self.hasImage():

            if event.key() == Qt.Key_Control and not self.global_erase_override:
                self._deleteCrossHandles[0].hide()
                self._deleteCrossHandles[1].hide()

            # Show the overlay again
            if event.key() == Qt.Key_H:
                self._overlayHandle.show()

        QGraphicsView.keyPressEvent(self, event)

    def mousePressEvent(self, event):
        
        

        if self.hasImage():
            """ Start drawing, panning with mouse, or zooming in
            """
            scenePos = self.mapToScene(event.pos())
            if event.button() == Qt.LeftButton:

                self._overlay_stack.append(self.mask_pixmap.copy())
                if self.direct_mask_paint:
                    self._offscreen_mask_stack.append(self._offscreen_mask.copy())

                # If ALT is held, replace color
                repaint_was_active = False
                if QApplication.keyboardModifiers() & Qt.AltModifier:
                    try:
                        repaint_was_active = True
                        self.viewport().setCursor(Qt.BusyCursor)
                        self.repaintArea()
                    except Exception as e:
                        print("Cannot repaint region. Additional information:")
                    self.viewport().setCursor(Qt.ArrowCursor)

                # If SHIFT is held, draw a line
                if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                    self.drawMarkerLine(event)

                # If CONTROL is held, erase, but only if global erase override is not enabled
                if not self.global_erase_override:
                    if QApplication.keyboardModifiers() & Qt.ControlModifier:
                        self.current_painting_mode = self.MODE_ERASE
                    else:
                        self.current_painting_mode = self.MODE_PAINT

                # If the user just clicks, add a marker (unless repainting was done)
                if not repaint_was_active:
                    print("fillMarker")
                    #self.fillMarker(event)

                self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
            elif event.button() == Qt.MiddleButton:
                if self.canPan:
                    self.__prevMousePos = event.pos()
                    self.viewport().setCursor(Qt.ClosedHandCursor)
                self._cursorHandle.hide()
                self.middleMouseButtonPressed.emit(scenePos.x(), scenePos.y())
            elif event.button() == Qt.RightButton:

                if self.canZoom:
                    self.setDragMode(QGraphicsView.RubberBandDrag) # self.numbox
                self._cursorHandle.hide()
                self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())                
                
                self.selection = [event.pos()] # ai mask - 2 method
        
        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """ Stop mouse pan or zoom mode (apply zoom if valid).
        """
        if self.hasImage():
            QGraphicsView.mouseReleaseEvent(self, event)
            scenePos = self.mapToScene(event.pos())
            if event.button() == Qt.MiddleButton:
                self.viewport().setCursor(Qt.ArrowCursor)
                self._cursorHandle.show()
                self.middleMouseButtonReleased.emit(scenePos.x(), scenePos.y())
            elif event.button() == Qt.RightButton:
                
                '''
                if self.canZoom:
                    viewBBox = self.zoomStack[-1] if len(self.zoomStack) else self.sceneRect()
                    selectionBBox = self.scene.selectionArea().boundingRect().intersected(viewBBox)
                    self.scene.setSelectionArea(QPainterPath())  # Clear current selection area.
                    if selectionBBox.isValid() and (selectionBBox != viewBBox):
                        self.zoomStack.append(selectionBBox)
                        self.updateViewer()
                '''
                self.setDragMode(QGraphicsView.NoDrag)
                self._cursorHandle.show()
                self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())
                if self.isANDImethod == True:
                    if self.numbox > 0: 
                        self.numbox = 0
                        self.min_x_2, self.max_x_2, self.min_y_2, self.max_y_2 = self.min_x, self.max_x, self.min_y, self.max_y
                        self.isANDImethod = False
                    else:
                        self.numbox =+ 1
                        self.min_x_1, self.max_x_1, self.min_y_1, self.max_y_1 = self.min_x, self.max_x, self.min_y, self.max_y


                else: self.numbox = 0
                
        QGraphicsView.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        if self.hasImage():
            scenePos = self.mapToScene(event.pos())
            if event.button() == Qt.MiddleButton:
                self.middleMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
            elif event.button() == Qt.RightButton:
                if self.canZoom:
                    self.zoomStack = []  # Clear zoom stack.
                    self.updateViewer()
                self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    def handleMiddleClick(x, y):
        row = int(y)
        column = int(x)

    # Create the application.
    app = QApplication(sys.argv)

    # Create image viewer and load an image file to display.
    viewer = QtImageAnnotator()
    #viewer.loadImageFromFile()  # Pops up file dialog.

    # Show viewer and run application.
    viewer.show()
    sys.exit(app.exec_())
