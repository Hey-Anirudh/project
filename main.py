import cv2
import numpy as np
from utils.hand_detector import HandDetector
from utils.canvas import VirtualCanvas

class HandWritingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)  # Width
        self.cap.set(4, 720)   # Height
        
        self.detector = HandDetector(max_hands=1)
        self.canvas = VirtualCanvas()
        
        # Colors for UI
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 255, 255) # White
        ]
        
        self.color_names = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "White"]
        self.selected_color = 1  # Start with green
        
        self.brush_sizes = [3, 5, 7, 10, 15]
        self.selected_brush = 2  # Medium brush
        
        # Control states
        self.drawing = False
        self.show_menu = True
        self.ui_overlay = np.zeros((720, 300, 3), dtype=np.uint8)
        
    def draw_ui(self, img):
        """Draw the control panel UI"""
        if not self.show_menu:
            return
        
        overlay = self.ui_overlay.copy()
        
        # Title
        cv2.putText(overlay, "Hand Writing App", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Color selection
        cv2.putText(overlay, "Colors:", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        y_offset = 120
        for i, (color, name) in enumerate(zip(self.colors, self.color_names)):
            color_rect = (20, y_offset + i*40, 30, 30)
            cv2.rectangle(overlay, color_rect, color, -1)
            if i == self.selected_color:
                cv2.rectangle(overlay, color_rect, (255, 255, 255), 2)
            cv2.putText(overlay, name, (70, y_offset + i*40 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Brush sizes
        cv2.putText(overlay, "Brush Size:", (20, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        y_offset = 430
        for i, size in enumerate(self.brush_sizes):
            cv2.circle(overlay, (40, y_offset + i*40), size, (255, 255, 255), -1)
            if i == self.selected_brush:
                cv2.circle(overlay, (40, y_offset + i*40), size + 2, (0, 255, 0), 2)
            cv2.putText(overlay, f"Size {size}", (70, y_offset + i*40 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(overlay, "Controls:", (20, 650), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(overlay, "Index: Draw/Erase", (20, 680), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, "Thumb+Index: Select", (20, 700), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend UI with main image
        img[:, :300] = cv2.addWeighted(img[:, :300], 0.3, overlay, 0.7, 0)
        
    def check_ui_interaction(self, x, y):
        """Check if user is interacting with UI elements"""
        if x < 300 and self.show_menu:
            # Color selection
            if 120 <= y <= 320:
                color_index = (y - 120) // 40
                if 0 <= color_index < len(self.colors):
                    self.selected_color = color_index
                    self.canvas.set_color(self.colors[color_index])
                    return True
            
            # Brush size selection
            if 430 <= y <= 630:
                brush_index = (y - 430) // 40
                if 0 <= brush_index < len(self.brush_sizes):
                    self.selected_brush = brush_index
                    self.canvas.set_brush_size(self.brush_sizes[brush_index])
                    return True
        
        return False
    
    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
                
            img = cv2.flip(img, 1)
            img = self.detector.find_hands(img)
            
            # Get hand landmarks
            landmarks = self.detector.find_position(img, draw=False)
            
            if len(landmarks) != 0:
                # Get index and thumb tip positions
                index_tip = self.detector.get_finger_tip(img, 8)
                thumb_tip = self.detector.get_finger_tip(img, 4)
                
                if index_tip[0] is not None and thumb_tip[0] is not None:
                    index_x, index_y = index_tip
                    thumb_x, thumb_y = thumb_tip
                    
                    # Calculate distance between thumb and index
                    distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
                    
                    # Check if thumb and index are close together (selection gesture)
                    if distance < 50:
                        if not self.check_ui_interaction(index_x, index_y):
                            self.drawing = False
                            self.canvas.reset_previous_point()
                    else:
                        # Check if index finger is up for drawing
                        if (self.detector.is_finger_up(img, 8, 6) and  # Index finger
                            not self.detector.is_finger_up(img, 12, 10)):  # Middle finger
                            
                            if index_x > 300:  # Don't draw on UI area
                                self.drawing = True
                                self.canvas.add_point(index_x, index_y)
                            else:
                                self.drawing = False
                                self.canvas.reset_previous_point()
                        else:
                            self.drawing = False
                            self.canvas.reset_previous_point()
                    
                    # Check middle finger for eraser mode toggle
                    if (self.detector.is_finger_up(img, 12, 10) and  # Middle finger
                        not self.detector.is_finger_up(img, 8, 6)):   # Index finger down
                        self.canvas.toggle_mode()
                        self.canvas.reset_previous_point()
                        self.drawing = False
                        cv2.waitKey(300)  # Debounce
            else:
                self.drawing = False
                self.canvas.reset_previous_point()
            
            # Draw UI
            self.draw_ui(img)
            
            # Get canvas and overlay it on camera feed
            canvas_img = self.canvas.get_canvas()
            
            # Create mask for canvas (where there's drawing)
            mask = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            
            # Apply canvas to camera feed
            img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(canvas_img, canvas_img, mask=mask)
            img = cv2.add(img_bg, canvas_fg)
            
            # Display mode indicator
            mode_text = f"Mode: {self.canvas.mode.capitalize()}"
            cv2.putText(img, mode_text, (320, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Clear canvas with pinky finger gesture
            if len(landmarks) != 0:
                if (self.detector.is_finger_up(img, 20, 18) and  # Pinky finger
                    self.detector.is_finger_up(img, 16, 14) and  # Ring finger
                    self.detector.is_finger_up(img, 12, 10)):    # Middle finger
                    self.canvas.clear_canvas()
                    cv2.waitKey(300)  # Debounce
            
            # Toggle UI with all fingers up
            if len(landmarks) != 0:
                all_fingers_up = all([
                    self.detector.is_finger_up(img, 8, 6),   # Index
                    self.detector.is_finger_up(img, 12, 10), # Middle
                    self.detector.is_finger_up(img, 16, 14), # Ring
                    self.detector.is_finger_up(img, 20, 18)  # Pinky
                ])
                if all_fingers_up:
                    self.show_menu = not self.show_menu
                    cv2.waitKey(500)  # Debounce
            
            cv2.imshow("Hand Writing App", img)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas.clear_canvas()
            elif key == ord('m'):
                self.canvas.toggle_mode()
            elif key == ord('u'):
                self.show_menu = not self.show_menu
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandWritingApp()
    app.run()