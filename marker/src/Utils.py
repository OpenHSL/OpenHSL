class Utils:

    @staticmethod
    def is_hex_color(string):
        string = string.lstrip('#')
        if len(string) == 3 or len(string) == 6:
            try:
                _ = int(string, 16)
                return True
            except ValueError:
                return False
        return False

    @staticmethod
    def hex_to_rgb(string):
        string = string.lstrip('#')
        if len(string) == 3:
            r = int(string[0], 16)
            g = int(string[1], 16)
            b = int(string[2], 16)
        elif len(string) == 6:
            r = int(string[0:2], 16)
            g = int(string[2:4], 16)
            b = int(string[4:6], 16)
        return r, g, b

    @staticmethod
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    @staticmethod
    def average_hex_colors(hex1, hex2):
        if Utils.is_hex_color(hex1) and Utils.is_hex_color(hex2):
            r1, g1, b1 = Utils.hex_to_rgb(hex1)
            r2, g2, b2 = Utils.hex_to_rgb(hex2)
            r = (r1 + r2) // 2
            g = (g1 + g2) // 2
            b = (b1 + b2) // 2
            avg_hex = Utils.rgb_to_hex((r, g, b))
            return avg_hex
        return None

    @staticmethod
    def clamp(val, a, b):
        if a < b:
            return max(a, min(val, b))
        else:
            return max(b, min(val, a))

    @staticmethod
    def do_nothing():
        print("do nothing")
