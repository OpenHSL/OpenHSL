# datm-annotation-tool
datm-annotation-tool is a PyQt5 application intended for annotating road segment orthoframes in two ways:

* Marking pavement defects by painting over them with different colors depending on the type of defect;
* Masking out the paved area of the road, this usually means to correct an existing mask.

Both of these goals are achieved using painting tools implemented in a standalone component **QtImageAnnotator** derived from [PyQtImageViewer](https://github.com/marcel-goldschen-ohm/PyQtImageViewer). This component can be used separately from the application. It is available in the `ui_lib` folder.

Note that for large images (over 4k by 4k resolution in either dimension) the painting will become slower.

Basic instructions on how to use the tool are provided next.

### New in Version 1.0

Version 1.0 marks the transition from single color defect in-painting to support different colors for different defects. This relationship is set up in the `.csv` file that can be found under `defs/color_defs.csv`.  Accordingly, some productivity features were introduced to help with the annotation workflow:

* The list of colors has been added according to the specifications. It can be viewed also by going to **View→Color specifications**. The types of defects currently annotated can be switched quickly from the keyboard by **pressing the corresponding number keys** listed in the color specification dialog box.
* Defect masks are now exported in grayscale based on a mapping defined in the color specification. No changes in road area masking process.
* It is possible to **repaint** a connected contour by alt-left clicking over the pixels of the desired color. It can be useful for taking “undefined” defects and repainting them. Note however that at the moment, if other defects are connected to the repainted one with at least 1 pixel, the whole contour will be repainted.
* **[CTRL]+[X]** over a connected contour of the current paint color will remove that contour. Tip: to quickly remove a contour that is connected to something else, **[SHIFT]+left click** along the border of the contour to disconnect it from the surrounding contours, and then use this feature on the contour.
* **[CTRL]+[Q]** over a connected contour of any color or a mix of colors will remove it completely irrespective of colors.
* The TK layer, if available, can be toggled on and off using **[T]** shortcut key on the keyboard.
* There should be full backwards-compatibility with the previous version (old masks can be loaded). However, the new “undefined” defect color is white.
* Some redundant features (mostly useless menu entries) were removed.

### Installation

The most common way to install the tool is to use an Anaconda Python environment. Assuming Anaconda is not installed, these are the steps to follow:

1. Install the latest Anaconda Python 3.7+ distribution from [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
2. Clone or download *datm-annotation-tool* repository from GitHub.
3. Open *Anaconda Prompt* and `cd` to the root of the downloaded repository.
4. Run `conda env create -f datm-annotation-tool.yml` and wait for the process to finish.
5. Run `conda activate datm-annotation-tool`. Note that you will have to activate the environment every time you want to run the application from Anaconda Prompt.
6. Finally, run the application with `python datmant.py`
7. **NB!** You can also build an executable version of the application with the provided batch script (on Windows only) by running `Build_Win64_executable` from Anaconda Prompt in the repository directory after following steps 1 through 5. In this case, the `datmant` folder will appear under the  `dist` folder in the repository root. In it, you will find the `datmant.exe` which you can now use to start the application. You can copy the `datmant` folder to any desired location on your hard drive or even share with your colleagues who do not have Python installed on Windows. Note that the resulting folder can have a size of about 1GB uncompressed.

### Usage

#### File System Considerations

After starting the tool, you are presented with the following interface:



![LaunchWindow](.github/img/main_gui_empty.png)



The first thing to do, is to browse to the folder that contains the orthoframes of interest. To do this, click on the **Browse...** button in the bottom portion of the interface. Note, that every orthoframe image `FILENAME.jpg` file in the folder is assumed to have particular companion files:

* `FILENAME.mask.png`: the initial mask of the paved road part of the image. If it does not exist, an empty mask will be used.
* `FILENAME.vrt`: file with image geometry data (optional): necessary for marking defects if they are already stored in a relevant shapefile database (see below).
* `FILENAME.predicted_defects.png` (optional): automatically generated defect masks to be manually processed (if available).

Secondly, if preprocessed “Tehnokeskuse” (TK) defect layers are present, then Defect .shp folder should be selected by clicking the corresponding **Browse...** button and selecting the folder. Note that it is assumed the folder has the following types of shapefiles (`.shp`) with their support files:

* `defects_line.*`: defects marked as lines;
* `defects_point.*`: defects marked as points;
* `defects_polygon.*`: defects marked as polygons.

Once the folder with orthoframes of interest is selected, the first orthoframe found in the folder will be automatically loaded into the tool:



![LaunchWindow](.github/img/main_gui.png)



You can expand the tool by either manually resizing its window or by double-clicking the title bar and begin the annotation process.

The tool will produce the following files for every processed orthoframe:

* `FILENAME.cut.mask_v2.png`: The manually corrected mask (usually some manual correction is required). If no correction is made to the original mask, this file will contain a copy of the original mask.
* `FILENAME.defect.mask.png`: The mask for defects found on the orthoframe.

#### Annotating Orthoframes: Tutorial

As discussed above, the objective is to create two masks for each orthoframe of interest. First one is intended for marking defects, and the second one for correcting the paved road area mask.

The topmost graphics port of the application is used to paint in the defects and mask once an image is loaded into it. By default, the *defect marking mode* is always set when a new image is loaded into memory.

It is assumed that a typical computer mouse with three buttons and a scroll wheel is used for manipulating the image.

**Painting tools:**

* To change the painting mode (defects or mask in-painting), use the button on the bottom of the UI. The color of the button also hints at the colors with which defects and non-paved areas are marked: blue and red, respectively.
* To change the  type of defect that is being annotated, click on the **Annotated defect**  combo box and choose the correct defect type. Alternatively, you can use number shortcuts on the keyboard: **[1]** will select the first defect type, **[2]** the second defect type and so on until **[9]**.
* Color specifications can be accessed from the menu by going to View→Color specifications. A pop up window will appear.
* To paint over a defect or unpaved area (depending on the mode), **left click and drag.**
* To erase any paint (any annotation is erased independent on the mode!), either switch to **delete mode** by pressing **[D]** on the keyboard, or use **[CTRL]-left click** in normal painting mode. Either way, you will see an X painted over the brush cursor that symbolizes that you have entered delete mode.
* To temporarily hide annotations, press and hold the **[H]** key on the keyboard.
* To toggle the display of the TK defect layer, press **[T]** key on the keyboard.
* To change brush size, **rotate the mouse wheel** while **holding [CTRL]**. You can also change the size of the brush using the corresponding slider in the UI.
* To create *line segments*, **left click** once at the starting point and then **[SHIFT]-left click** at the end point.
* To fill bounded areas with selected brush color, position the brush over an empty area and press **[F]** on the keyboard. The currently selected color will be used. Note that other color strokes will block the fill.
* To remove a connected contour having the current color, hover over the contour you wish to remove and press **[CTRL]+[X]**.
* To remove a connected contour regardless of color, hover over the contour and press **[CTRL]+[Q]**.
* To repaint a connected contour regardless of color, hover over the contour and use **[ALT]-left click**.
* There is also a 20 step buffer for undoing incorrect painting operations via the usual shortcut **[CTRL]+[Z]**. Note however that a forward step is not yet available, so once undone you cannot reverse the operation via [CTRL]+[Y].

At the moment, the *Clear ALL annotations* feature will also clear the undo buffer. Because this operation is highly destructive, the **[R]** key shortcut has been removed.

**Navigation tools:**

* To zoom in on a particular portion of the image **right click and drag** over the region of interest.
* To zoom out, **double right click**.
* Use the **[+]** and **[−]** keys on the keyboard to zoom in and out. Zooming in will take into account the last known cursor location and magnification will be applied in discrete steps.
* To pan the image while zoomed-in, **middle click and drag**.

Note that **companion files will be automatically saved for each orthoframe only** once you choose another orthoframe from the *Current image* list or press **[P]** *Previous image* or **[N]** *Next image* or choose **File→Save current annotations** from the menu. The application also warns you when navigating files whether you have reached either end of the folder.

What concerns different views and mask generation, you have the options described below.

**NB! Changing these options will result in you losing any current defect annotations unless you save them beforehand, so if you want to keep the annotations of defects, you need to save them using File→Save current annotations**

* The **Edit→Process original mask** checkbox applies preprocessing to the original mask which can speed up the mask correction workflow. Note **you will only see the original/preprocessed mask** if the corrected mask `FILENAME.cut.mask_v2.png` is **not** found in the folder. If you would like to restart the mask correction process, please manually delete the corresponding file taking note of the current image `FILENAME`.
* The **Edit→Reload AUTO defect mask** menu entry reloads the automatically generated defect mask if it is present in the working directory and the annotation mode is set to defect annotation.

The procedure for in-painting defects and correcting the mask is showcased for a single orthoframe in the following animation. NB! This is not meant to be an instructional video on how to correctly paint in the defects, just an example of using the application.

![LaunchWindow](.github/img/datmant_usage.gif)
