import cv2
import numpy as np

class VirtualCanvas:
    def __init__(self, width=1280, height=720):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.width = width
        self.height = height
        self.previous_point = None
        self.drawing_color = (0, 255, 0)  # Green color
        self.brush_size = 7
        self.eraser_size = 20
        self.mode = "draw"  # "draw" or "erase"
        
    def draw_line(self, x1, y1, x2, y2):
        if self.mode == "draw":
            cv2.line(self.canvas, (x1, y1), (x2, y2), self.drawing_color, self.brush_size)
        elif self.mode == "erase":
            cv2.circle(self.canvas, (x2, y2), self.eraser_size, (0, 0, 0), -1)
    
    def add_point(self, x, y):
        if self.previous_point:
            x1, y1 = self.previous_point
            self.draw_line(x1, y1, x, y)
        self.previous_point = (x, y)
    
    def reset_previous_point(self):
        self.previous_point = None
    
    def clear_canvas(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def set_color(self, color):
        self.drawing_color = color
    
    def set_brush_size(self, size):
        self.brush_size = size
    
    def set_eraser_size(self, size):
        self.eraser_size = size
    
    def toggle_mode(self):
        self.mode = "erase" if self.mode == "draw" else "draw"
    
    def get_canvas(self):
        return self.canvas.copy()