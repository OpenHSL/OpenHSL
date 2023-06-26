class Layer:
    def __init__(self, json_layer, cv_mask):
        self.name = json_layer["name"]
        self.color = json_layer["color"]
        self.isVisible = json_layer["visible"]
        self.isLocked = json_layer["locked"]

        self.cvMask = cv_mask
