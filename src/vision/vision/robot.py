class Robot:
    def __init__(self, id, x, y, orientation, port):
        self.id = id
        self.x = x
        self.y = y
        self.orientation = orientation
        self.port = port

    def update(self, x, y, orientaion):
        """
        Updates the robot's position and orientation.
        
        Args:
            x (float): New X-coordinate.
            y (float): New Y-coordinate.
            orientation (float): New orientation in radians.
        """
        self.x = x
        self.y = y
        self.orientation = orientaion

    def identify(self):
        print(f"Robot {self.id} on ({self.x}, {self.y}), orientation: {self.orientation}, port: {self.port}")

    # def send_data(self, )
