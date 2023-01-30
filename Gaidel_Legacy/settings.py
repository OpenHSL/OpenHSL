# -*- coding: utf-8 -*-
import configparser
import os
import yaml


MODE_PLAIN = "plain"
# MODE_VIDEO = "video"
# MODE_WORKER = "worker"
# MODE_IMAGE_SAVING = [MODE_PLAIN, MODE_VIDEO, MODE_WORKER]


class Config(object):

    def __init__(self, config_dict):
        self.host = config_dict.get("host", "127.0.0.1")
        self.path = config_dict.get("path", "/DCIM/100MEDIA/")
        self.password = config_dict.get("password", "")
        self.port = config_dict.get("port", "")
        self.protocol = config_dict.get("protocol", "http")
        self.user = config_dict.get("user", "user")
        self.enable_gps_log = config_dict.get("enable_gps_log", True)
        self.enable_mavros = config_dict.get("enable_mavros", True)
        self.enable_picamera = config_dict.get("enable_picamera", True)
        self.enable_rest_server = config_dict.get("enable_rest_server", True)
        self.enable_tiscam = config_dict.get("enable_tiscam", True)
        self.enable_web_interface = config_dict.get("enable_web_interface", True)
        self.enable_web_video_server = config_dict.get("enable_web_video_server", True)
        self.image_saving_mode = config_dict.get("image_saving_mode", MODE_PLAIN)
        #self.postprocessing_fix_vignetting = config_dict.get("postprocessing_fix_vignetting", FIX_VIGNETTING)
        #self.postprocessing_scaling_gray_levels = config_dict.get(
        #    "postprocessing_scaling_gray_levels", SCALING_GRAY_LEVELS
        #)
        self.spectral_bands_number = config_dict.get("spectral_bands_number", BANDS_NUMBER)

    def __len__(self):
        return len(self.__dict__)

    #def update(self):
    #    with open(os.path.join(PROJECT_DIR, FILENAME_YAML), MODE_READ) as stream:
    #        self.__dict__.update(yaml.load(stream, Loader=yaml.FullLoader))


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# ACTION_STORE_TRUE = "store_true"
# APPEND_MODE = "a"
BANDS_NUMBER = 40
# BLUR_AUTO = False
BLUR_SHAPE = (3, 3)
BORDER_TOP = 0.0
BORDER_BOTTOM = 1.0
# CAMERA_INDEX = -1
CAMERA_PITCH = 0.0
CAMERA_TANGENT = 0.30
# COLOR_MAX = 255
# COMMAND_CONVERT_VIDEO = "MP4Box -add {input} {output}"
# COMMAND_MAVROS_UART = ["roslaunch", "mavros", "px4.launch", 'fcu_url:=/dev/ttyAMA0:921600']
# COMMAND_MAVROS_UDP = ["roslaunch", "mavros", "px4.launch", 'fcu_url:=udp://@192.168.0.141:14580']
# COMMAND_MIDDLEWARE = ["python3", "-m", "middleware.FlaskServer"]
# COMMAND_NPM = ["npm", "run", "serve", "--", "--port", "8001"]
# COMMAND_PYTHON = ["python3", "-m", "python_module.FlaskServer"]
# COMMAND_WEB_VIDEO_SERVER = ["rosrun", "web_video_server", "web_video_server"]
CSV_DELIMITER = ";"
# DETECT_RAINBOW = False
# DIR_GPS = "gps"
# DIR_PICAMERA = "picamera"
# DIR_TCAM = "tcam"
# DIR_TEMP = "tmp"
# DIR_VIDEO = "video"
# DIR_RECORDING = "results"
DISTANCE_LIMIT = 1.0
# EXTENSION_JSON = "json"
# EXTENSION_NUMPY = "npy"
# EOL = "\n"
# EPS = 1.0e-15
# EVENT_STATUS = "status"
EXIT_KEY = "q"
# FILENAME_DATA = os.path.join(PROJECT_DIR, "data", "config.json")
# FILENAME_HYPER_VIDEO_PATTERN = "rec_*.avi"
# FILENAME_IMAGE_EXTENSION_JPEG = ".jpg"
# FILENAME_IMAGE_PATTERN = "{name}-{frame}{extension}"
# FILENAME_GPS_PATTERN = "gps-%Y-%m-%d-%H-%M-%S.csv"
# FILENAME_POSITION = "position.csv"
# FILENAME_TIME_PATTERN = "%Y-%m-%d-%H-%M-%S"
# FILENAME_VIDEO_EXTENSION_AVI = ".avi"
# FILENAME_VIDEO_EXTENSION_H264 = ".h264"
# FILENAME_VIDEO_EXTENSION_MP4 = ".mp4"
FILENAME_YAML = "settings.yaml"
# FIX_VIGNETTING = False
# FORMAT_FULL_TIME = "%Y-%B-%d.%m.%Y-%H:%M"
# FORMAT_TIME = "%H:%M"
# FPS = 25
# GAIN_ANALOG = 1.0
# GAIN_AWB = None
# GAIN_DIGITAL = 1.0
# GPIO_PIN_DEFAULT = 14
# GPIO_PWM_FREQUENCY = 50
# GPIO_PWM_MIN_DUTY_CYCLE = 1.0
# GPIO_PWM_MAX_DUTY_CYCLE = 12.0
# GPIO_PWM_STEP_DUTY_CYCLE = 0.1
GPS_HYPERCAM_FRAME = "Hypercam frame"

# GST_PIPELINE_PATTERN = (
#     "tcambin name={name}"
#     " ! videoconvert"
#     " ! appsink name={sink}"
# )
# GST_PIPELINE_VIDEO_PATTERN = (
#     "tcambin name={name}"
#     " ! queue"
#     " ! videoconvert"
#     " ! x264enc"
#     " ! avimux"
#     " ! filesink location={filename}"
# )
HEADER_CAM_ID = "cam_ID"
# HEADER_LATITUDE = "latitude"
# HEADER_LONGITUDE = "longitude"
HEADER_REL_ALT = "rel_alt"
HEADER_ANGLE = "compass_hdg"
HEADER_X = "x"
HEADER_Y = "y"
HEADER_Z = "z"
# IMAGE_PATTERN = "img-{0:06d}.jpg"
# ISO = 200
# LIMIT_QUEUE_SIZE = 1000
# LOG_DIR = "log"
# LOG_FILENAME = "log.txt"
# MAVROS_DEVICE = "/dev/ttyAMA0"
# MODE_AUTO = "auto"
# MODE_AWB = "off"
# MODE_EXPOSURE = "off"
# MODE_FLASH = "off"
MODE_READ = "r"
# MODE_WRITE = "w"
# NETWORK_BROADCAST = "0.0.0.0"
# NETWORK_LOCALHOST = "127.0.0.1"
# PATH_PATTERN = "{protocol}://{user}:{password}@{host}:{port}/video"
PROPERTIES_FILE = "settings.ini"
# PYTHON_PATH = "/home/pi/workspace/hyperspectralwebanalyze"
# REPORT_FRAMES = 100
# REPORT_STATUS = True
# RESCALE_AUTO = False
# REST_PORT = 8000
# RESULTS_DIR = "results"
# ROS_PI_CAMERA_NAME = "raspicam"
# ROS_TIS_CAMERA_NAME = "tiscam"
# SCALING_GRAY_LEVELS = False
# SEED = (hash("I feel lucky!") + (2 ** 32)) % (2 ** 32)
# SHAPE_OUTPUT = (1280, 720)
# SPECTRAL_FLIP = True
# SPEED_EXPOSURE = None
# SPEED_SHUTTER = None
# TARGET_ENCODING = "utf_8"
# PATH_WEB_INTERFACE = "/home/pi/workspace/hyperspectralwebanalyze/web_module/hyperspectral-web"
# PICAMERA_BRIDGE_MODE = "bgr8"
# PICAMERA_DISPLAY_SHAPE = (832, 624)
# PORT_MIDDLEWARE = 51116
# PORT_PYTHON = 51117
# PORT_STATUS = 51117
# PRINCIPAL_SLICES = False
# ROTATE_ORIGIN = 0
# ROTATE_TIMES = 0
TARGET_RESOLUTION = 1080
# # TARGET_RESOLUTION = 3840
# TCAM_BRIDGE_MODE = "mono8"
# TCAM_EXPOSURE = 20000
# TCAM_GAIN = 16
# TCAM_FORMAT = "BGRx"
# TCAM_FPS = 60
# TCAM_FSINK = "fsink"
# TCAM_HEIGHT = 480
# TCAM_NAME = "source"
# TCAM_SINK = "sink"
# TCAM_TARGET_HEIGHT = 480
# TCAM_TARGET_WIDTH = 640
# TCAM_WIDTH = 744
# TIME_CONFIGURING = 1.0
# TIME_ROTATING = 1.0
# TIMEOUT = 10.0
# TIMEOUT_QUEUE_GET = 1.0
# URL_LIST_PATTERN = "{protocol}://{host}{path}"
# URL_WEB_INTERFACE = "http://localhost:51117/flights"
# VERBOSE = False
# VERTICAL_FLIP = True
# HORIZONTAL_FLIP = False
# VIDEO_CODEC_FOURCC = "mp4v"
WAVELENGTH_MIN = 400.0
WAVELENGTH_MAX = 1100.0
# ZERO_FILL = 6

config = None


def update():
    global config
    parser = configparser.ConfigParser()
    parser.read(os.path.join(PROJECT_DIR, PROPERTIES_FILE))
    config = dict(parser["DEFAULT"])
    with open(os.path.join(PROJECT_DIR, FILENAME_YAML), MODE_READ) as config_stream:
        config_yaml = yaml.load(config_stream, Loader=yaml.FullLoader)
    config.update(config_yaml)
    if len(config) == 0:
        config = None
    else:
        config = Config(config)


update()
