import os
import Gaidel_Legacy.settings as settings


MILLIS_IN_SECOND = 1000.0


def make_path(config):
    return settings.PATH_PATTERN.format(protocol=config.protocol, user=config.user, password=config.password,
                                        host=config.host, port=config.port)


def clear_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def change_extension(filename, extension):
    return os.path.splitext(os.path.basename(filename))[0] + "." + extension