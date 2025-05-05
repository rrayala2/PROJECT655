import cv2
import numpy as np
import time
import os
import threading
import sys
import math
import collections
import pyautogui
import queue
import logging
from collections import deque
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

# Set PyAutoGUI to be more responsive with no pause between operations
pyautogui.PAUSE = 0
# Failsafe turned off for smoother operation - use with caution
pyautogui.FAILSAFE = False

# Try to import win32api for direct mouse control (more responsive)
try:
    import win32api
    import win32con
    import win32gui
    HAVE_WIN32API = True
except ImportError:
    HAVE_WIN32API = False
    print("win32api not available, falling back to PyAutoGUI for mouse control")

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("presentation_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PresentationController")

# Global flag for signaling program termination
exit_flag = False

# Constants for gesture recognition
SWIPE_THRESHOLD = 50  # Pixels (reduced for better sensitivity)
SWIPE_FRAMES = 5  # Number of frames to check for swipe
THUMBS_ANGLE_MIN = 30  # Degrees
THUMBS_ANGLE_MAX = 150  # Degrees

# Constants for mouse control
MOUSE_SMOOTHING_FACTOR = 0.3  # Lower value = smoother but more lag (0.0-1.0)
MOUSE_ACCELERATION = 1.5  # Higher value = more responsive to small movements
MOUSE_DEAD_ZONE = 5  # Pixels of movement to ignore (reduces jitter)

# Import MediaPipe
import mediapipe as mp


class LaserOverlay:
    """Creates a transparent overlay window to display the laser pointer on top of presentations."""
    
    def __init__(self, laser_color=(255, 0, 0), laser_size=10, trail_length=5):
        """
        Initialize the laser overlay.
        
        Args:
            laser_color: RGB color tuple for the laser pointer
            laser_size: Size of the laser dot in pixels
            trail_length: Number of positions to keep for trail effect
        """
        self.laser_color = laser_color
        self.laser_size = laser_size
        self.trail_length = trail_length
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Track position history for trail effect
        self.position_history = collections.deque(maxlen=trail_length)
        
        # Flag to control overlay visibility
        self.visible = False
        
        # Store the current laser position
        self.current_position = None
        
        # Annotation mode state
        self.annotation_mode = False
        self.annotation_points = []
        self.annotations = []  # List of completed annotations
        self.annotation_color = laser_color  # Use same color as laser
        self.annotation_thickness = 3
        
        # Add annotation timeout duration (3 seconds)
        self.annotation_duration = 3.0  # seconds
        
        # Create a queue for thread-safe operations
        self.command_queue = queue.Queue()
        
        # Start in a separate thread to avoid blocking
        self.init_thread = threading.Thread(target=self._initialize_tkinter)
        self.init_thread.daemon = True
        self.init_thread.start()
        
        print("Laser overlay initializing...")
    
    def _initialize_tkinter(self):
        """Initialize the Tkinter window (called in its own thread)"""
        # Initialize the overlay window
        self.root = tk.Tk()
        self.root.title("Laser Pointer Overlay")
        self.root.attributes("-topmost", True)  # Keep on top of all windows
        self.root.attributes("-transparentcolor", "black")  # Make black transparent
        self.root.attributes("-alpha", 0.8)  # Slight transparency for the laser
        
        # Make the window fullscreen and without decorations
        self.root.overrideredirect(True)  # Remove window decorations
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        
        # Create a canvas to draw on
        self.canvas = tk.Canvas(self.root, bg="black", 
                               width=self.screen_width, 
                               height=self.screen_height,
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Make the window click-through
        if HAVE_WIN32API:
            # Wait a moment for window to be ready
            self.root.update()
            # Get the window handle
            self.hwnd = win32gui.FindWindow(None, "Laser Pointer Overlay")
            # Set the window style to be click-through
            win32gui.SetWindowLong(
                self.hwnd,
                win32con.GWL_EXSTYLE,
                win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE) | 
                win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            )
        
        # Hide initially
        self.root.withdraw()
        
        # Start with a blank image
        self._update_overlay_internal(None)
        
        print("Laser overlay initialized")
        
        # Process command queue and update the UI
        self._process_events()
        
        # Start the Tkinter main loop (this will block this thread)
        self.root.mainloop()
    
    def show(self):
        """Queue a command to show the overlay."""
        self.command_queue.put(('show', None))
    
    def hide(self):
        """Queue a command to hide the overlay."""
        self.command_queue.put(('hide', None))
    
    def update_overlay(self, position):
        """Queue a command to update the overlay with a new position."""
        self.command_queue.put(('update', position))
    
    def start_annotation(self):
        """Start annotation mode and begin a new annotation."""
        self.command_queue.put(('start_annotation', None))

    def stop_annotation(self):
        """Stop annotation mode and save the current annotation."""
        self.command_queue.put(('stop_annotation', None))

    def add_annotation_point(self, position):
        """Add a point to the current annotation."""
        self.command_queue.put(('add_annotation_point', position))

    def clear_annotations(self):
        """Clear all annotations."""
        self.command_queue.put(('clear_annotations', None))
    
    def _show_internal(self):
        """Actually show the overlay (called on Tkinter thread)."""
        if not self.visible:
            self.visible = True
            self.root.deiconify()
            print("Laser overlay shown")
    
    def _hide_internal(self):
        """Actually hide the overlay (called on Tkinter thread)."""
        if self.visible:
            self.visible = False
            self.root.withdraw()
            print("Laser overlay hidden")
    
    def _start_annotation_internal(self):
        """Internal method to start annotation mode."""
        self.annotation_mode = True
        self.annotation_points = []
        print("Annotation mode started")

    def _stop_annotation_internal(self):
        """Internal method to stop annotation mode and save annotation."""
        if self.annotation_mode and len(self.annotation_points) > 1:
            # Save this annotation with a creation timestamp
            self.annotations.append({
                'points': self.annotation_points.copy(),
                'color': self.annotation_color,
                'thickness': self.annotation_thickness,
                'created_at': time.time()  # Add creation timestamp
            })
        
        self.annotation_mode = False
        self.annotation_points = []
        print("Annotation mode stopped")

    def _add_annotation_point_internal(self, position):
        """Internal method to add a point to the current annotation."""
        if self.annotation_mode and position:
            self.annotation_points.append(position)

    def _clear_annotations_internal(self):
        """Internal method to clear all annotations."""
        self.annotations = []
        print("Annotations cleared")
    
    def _update_overlay_internal(self, position):
        """
        Actually update the overlay (called on Tkinter thread).
        
        Args:
            position: (x, y) position on screen for the laser pointer
        """
        # Clear the canvas
        self.canvas.delete("all")
        
        # Get current time for annotation timeout
        current_time = time.time()
        
        # Create a new list for annotations that haven't timed out
        active_annotations = []
        
        # Draw all saved annotations that haven't timed out
        for annotation in self.annotations:
            points = annotation['points']
            color = annotation['color']
            thickness = annotation['thickness']
            created_at = annotation.get('created_at', current_time)  # Default to current time if not set
            
            # Check if annotation has timed out
            if current_time - created_at <= self.annotation_duration:
                # Keep this annotation for next frame
                active_annotations.append(annotation)
                
                # Convert RGB color to hex for tkinter
                hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                
                # Draw line segments connecting points
                if len(points) >= 2:
                    for i in range(1, len(points)):
                        p1 = points[i-1]
                        p2 = points[i]
                        
                        self.canvas.create_line(
                            p1[0], p1[1], p2[0], p2[1],
                            width=thickness,
                            fill=hex_color,
                            capstyle=tk.ROUND,
                            joinstyle=tk.ROUND
                        )
        
        # Replace annotations list with only the active ones
        self.annotations = active_annotations
        
        # Draw current annotation in progress
        if self.annotation_mode and len(self.annotation_points) >= 2:
            # Convert RGB color to hex for tkinter
            hex_color = f'#{self.annotation_color[0]:02x}{self.annotation_color[1]:02x}{self.annotation_color[2]:02x}'
            
            # Draw line segments for current annotation
            for i in range(1, len(self.annotation_points)):
                p1 = self.annotation_points[i-1]
                p2 = self.annotation_points[i]
                
                self.canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1],
                    width=self.annotation_thickness,
                    fill=hex_color,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND
                )
        
        # Skip drawing laser if position is None
        if position is None:
            return
        
        # Add to position history
        self.position_history.append(position)
        self.current_position = position
        
        # Add to annotation if in annotation mode
        if self.annotation_mode:
            self.annotation_points.append(position)
        
        # Convert RGB color to hex for tkinter
        hex_color = f'#{self.laser_color[0]:02x}{self.laser_color[1]:02x}{self.laser_color[2]:02x}'
        
        # Draw the trailing effect
        positions = list(self.position_history)
        if len(positions) >= 2:
            # Draw lines connecting positions with decreasing opacity
            for i in range(1, len(positions)):
                # Calculate opacity (decreases for older positions)
                opacity = 0.7 * (i / len(positions))
                
                # Calculate alpha value for hex color (255 is fully opaque)
                alpha = int(255 * opacity)
                
                # Get points for this segment
                p1 = positions[i-1]
                p2 = positions[i]
                
                # Draw line segment
                self.canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1],
                    width=max(1, int(self.laser_size * 0.3 * opacity)),
                    fill=hex_color
                )
        
        # Draw main laser dot
        self.canvas.create_oval(
            position[0] - self.laser_size, position[1] - self.laser_size,
            position[0] + self.laser_size, position[1] + self.laser_size,
            fill=hex_color, outline=""
        )
        
        # Add highlight effect for better visibility
        highlight_size = max(1, int(self.laser_size * 0.5))
        highlight_offset = max(1, int(self.laser_size * 0.3))
        highlight_x = position[0] - highlight_offset
        highlight_y = position[1] - highlight_offset
        
        self.canvas.create_oval(
            highlight_x - highlight_size, highlight_y - highlight_size,
            highlight_x + highlight_size, highlight_y + highlight_size,
            fill="white", outline=""
        )
    
    def _process_events(self):
        """Process commands from the queue and schedule the next check."""
        try:
            # Process all pending commands
            while not self.command_queue.empty():
                try:
                    command, data = self.command_queue.get_nowait()
                    
                    if command == 'show':
                        self._show_internal()
                    elif command == 'hide':
                        self._hide_internal()
                    elif command == 'update':
                        self._update_overlay_internal(data)
                    elif command == 'start_annotation':
                        self._start_annotation_internal()
                    elif command == 'stop_annotation':
                        self._stop_annotation_internal()
                    elif command == 'add_annotation_point':
                        self._add_annotation_point_internal(data)
                    elif command == 'clear_annotations':
                        self._clear_annotations_internal()
                    elif command == 'cleanup':
                        self.root.destroy()
                        return  # Exit the event loop
                    
                    self.command_queue.task_done()
                except queue.Empty:
                    break
            
            # Force update the overlay every frame to ensure annotations timeout properly
            # even when there are no other commands
            self._update_overlay_internal(self.current_position)
            
            # Schedule the next check
            self.root.after(16, self._process_events)  # ~60 fps (16ms)
        except tk.TclError:
            # Window was likely closed
            pass
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Schedule window destruction on the main thread
            self.command_queue.put(('cleanup', None))
            print("Laser overlay cleanup scheduled")
        except:
            pass


class GestureRecognizer:
    """Class for recognizing hand gestures from landmarks."""
    
    def __init__(self, history_size=5):  # Reduced from 10 to 5 for faster response
        """
        Initialize the gesture recognizer.
        
        Args:
            history_size: Number of frames to keep in history
        """
        # Gesture state machine states
        self.STATES = {
            'IDLE': 0,
            'POSSIBLE': 1,
            'CONFIRMED': 2,
            'COOLDOWN': 3
        }
        
        # Initialize gesture states
        self.gesture_states = {
            'swipe_left': self.STATES['IDLE'],
            'swipe_right': self.STATES['IDLE'],
            'thumbs_up': self.STATES['IDLE'],
            'thumbs_down': self.STATES['IDLE'],
            'index_pointing': self.STATES['IDLE'],  # Added for laser pointer control
            'pinch': self.STATES['IDLE']  # Add this for annotation feature
        }
        
        # Time of last detected gestures
        self.last_gesture_time = {gesture: 0 for gesture in self.gesture_states}
        
        # Cooldown period between gestures (seconds)
        self.gesture_cooldown = 0.3  # Reduced for better responsiveness
        
        # Counter for consecutive detections for reliability
        self.detection_counters = {gesture: 0 for gesture in self.gesture_states}
        self.min_consecutive_detections = 2  # Reduced for better responsiveness
        
        # History window for landmarks
        self.landmark_history = collections.deque(maxlen=history_size)
        
        # Debug mode for printing detection details
        self.debug = False
        
        # Add persistence tracking
        self.persistent_gestures = {}
        self.active_gestures = {}
        
        print("Gesture Recognizer initialized")
    
    def update_landmarks(self, landmarks):
        """
        Update landmark history with new landmarks.
        
        Args:
            landmarks: List of landmark points
        """
        if landmarks:
            # Add timestamp with landmarks for velocity calculations
            self.landmark_history.append({
                'landmarks': landmarks,
                'timestamp': time.time()
            })
    
    def get_palm_center(self, landmarks=None):
        """
        Calculate palm center from landmarks.
        
        Args:
            landmarks: Hand landmarks or None to use latest
            
        Returns:
            Tuple (x, y) of palm center or None
        """
        # Use provided landmarks or latest from history
        if landmarks is None:
            if not self.landmark_history:
                return None
            landmarks = self.landmark_history[-1]['landmarks']
        
        if len(landmarks) >= 21:  # MediaPipe hand landmarks
            # Use wrist and middle finger MCP for palm center
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            
            palm_x = (wrist[0] + middle_mcp[0]) // 2
            palm_y = (wrist[1] + middle_mcp[1]) // 2
            
            return (palm_x, palm_y)
        
        return None
    
    def get_index_fingertip(self, landmarks=None):
        """
        Get the position of the index fingertip.
        
        Args:
            landmarks: Hand landmarks or None to use latest
            
        Returns:
            Tuple (x, y) of index fingertip or None
        """
        # Use provided landmarks or latest from history
        if landmarks is None:
            if not self.landmark_history:
                return None
            landmarks = self.landmark_history[-1]['landmarks']
        
        if len(landmarks) >= 21:  # MediaPipe hand landmarks
            # Index fingertip is landmark 8
            return landmarks[8][:2]
        
        return None
    
    def calculate_vector(self, point1, point2):
        """
        Calculate vector from point1 to point2.
        
        Args:
            point1: Starting point (x, y)
            point2: Ending point (x, y)
            
        Returns:
            Vector as (dx, dy)
        """
        return (point2[0] - point1[0], point2[1] - point1[1])
    
    def calculate_magnitude(self, vector):
        """
        Calculate magnitude of a vector.
        
        Args:
            vector: Vector as (x, y)
            
        Returns:
            Magnitude as float
        """
        return math.sqrt(vector[0]**2 + vector[1]**2)
    
    def calculate_angle(self, vector1, vector2):
        """
        Calculate angle between two vectors in degrees.
        
        Args:
            vector1: First vector as (x, y)
            vector2: Second vector as (x, y)
            
        Returns:
            Angle in degrees
        """
        # Calculate dot product
        dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
        
        # Calculate magnitudes
        mag1 = self.calculate_magnitude(vector1)
        mag2 = self.calculate_magnitude(vector2)
        
        # Prevent division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
        
        # Calculate angle
        cos_angle = dot_product / (mag1 * mag2)
        
        # Clamp to valid range for arccos
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # Convert to degrees
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def detect_swipe(self):
        """
        Detect swipe gestures based on palm movement.
        
        Returns:
            'swipe_left', 'swipe_right', or None
        """
        # Need enough history for swipe detection
        if len(self.landmark_history) < 3:  # Reduced minimum needed
            return None
        
        # Get palm centers from recent frames
        palm_centers = []
        timestamps = []
        
        # Use all available history up to SWIPE_FRAMES
        for i in range(-min(SWIPE_FRAMES, len(self.landmark_history)), 0):
            landmarks = self.landmark_history[i]['landmarks']
            palm_center = self.get_palm_center(landmarks)
            if palm_center:
                palm_centers.append(palm_center)
                timestamps.append(self.landmark_history[i]['timestamp'])
        
        # Need at least 3 points
        if len(palm_centers) < 3:
            return None
        
        # Calculate total displacement
        start_point = palm_centers[0]
        end_point = palm_centers[-1]
        displacement = self.calculate_vector(start_point, end_point)
        
        # Absolute displacement in x direction
        abs_x_displacement = abs(displacement[0])
        
        # Check for significant horizontal movement (reduced threshold)
        if abs_x_displacement > SWIPE_THRESHOLD:
            # Check direction and ensure movement is mostly horizontal
            if abs(displacement[1]) < abs_x_displacement * 0.8:  # Allow more vertical movement
                if displacement[0] > 0:
                    return 'swipe_left'  # Moving right is swipe left command
                else:
                    return 'swipe_right'  # Moving left is swipe right command
        
        return None
    
    def detect_thumbs_gesture(self, landmarks):
        """
        Detect thumbs up/down gestures.
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            'thumbs_up', 'thumbs_down', or None
        """
        if not landmarks or len(landmarks) < 21:
            return None
        
        # Get thumb landmarks
        thumb_tip = landmarks[4][:2]  # Thumb tip
        thumb_ip = landmarks[3][:2]   # Thumb IP joint
        thumb_mcp = landmarks[2][:2]  # Thumb MCP joint
        wrist = landmarks[0][:2]      # Wrist
        
        # Get other fingertips
        index_tip = landmarks[8][:2]
        middle_tip = landmarks[12][:2]
        ring_tip = landmarks[16][:2]
        pinky_tip = landmarks[20][:2]
        
        # Simple thumb up: thumb is significantly above wrist, other fingers are lower than thumb
        if (thumb_tip[1] < wrist[1] - 30 and  # Thumb is above wrist (Y decreases upward)
            thumb_tip[1] < index_tip[1] and
            thumb_tip[1] < middle_tip[1] and
            thumb_tip[1] < ring_tip[1] and
            thumb_tip[1] < pinky_tip[1]):
            return 'thumbs_up'
        
        # Simple thumb down: thumb is significantly below wrist, other fingers are higher than thumb
        if (thumb_tip[1] > wrist[1] + 30 and  # Thumb is below wrist
            thumb_tip[1] > index_tip[1] and
            thumb_tip[1] > middle_tip[1] and
            thumb_tip[1] > ring_tip[1] and
            thumb_tip[1] > pinky_tip[1]):
            return 'thumbs_down'
        
        return None
    
    def detect_index_pointing(self, landmarks):
        """
        Detect if index finger is extended for pointing (laser pointer control).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if index finger is extended for pointing, False otherwise
        """
        if not landmarks or len(landmarks) < 21:
            return False
        
        # Get fingertips
        index_tip = landmarks[8][:2]
        middle_tip = landmarks[12][:2]
        ring_tip = landmarks[16][:2]
        pinky_tip = landmarks[20][:2]
        
        # Get index finger joints
        index_pip = landmarks[6][:2]  # Second joint from tip
        index_mcp = landmarks[5][:2]  # Knuckle
        
        # Get wrist position
        wrist = landmarks[0][:2]
        
        # Check if index finger is extended and other fingers are curled
        # Index finger should be significantly extended
        index_extended = self.calculate_magnitude(self.calculate_vector(index_mcp, index_tip)) > 50
        
        # Middle, ring, and pinky should be curled (closer to palm than their MCPs)
        middle_mcp = landmarks[9][:2]
        ring_mcp = landmarks[13][:2]
        pinky_mcp = landmarks[17][:2]
        
        # Simplified check: Index finger is extended while other fingers are more curled
        if index_extended:
            # Calculate distance from fingertips to wrist
            index_dist = self.calculate_magnitude(self.calculate_vector(wrist, index_tip))
            middle_dist = self.calculate_magnitude(self.calculate_vector(wrist, middle_tip))
            ring_dist = self.calculate_magnitude(self.calculate_vector(wrist, ring_tip))
            pinky_dist = self.calculate_magnitude(self.calculate_vector(wrist, pinky_tip))
            
            # Index should be furthest from wrist
            return index_dist > middle_dist and index_dist > ring_dist and index_dist > pinky_dist
        
        return False
    
    def detect_pinch(self, landmarks):
        """
        Detect index-thumb pinch gesture for annotation.
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if pinch gesture detected, False otherwise
        """
        if not landmarks or len(landmarks) < 21:
            return False
        
        # Get thumb and index fingertips
        thumb_tip = landmarks[4][:2]  # Thumb tip
        index_tip = landmarks[8][:2]  # Index fingertip
        
        # Calculate distance between thumb and index fingertips
        distance = self.calculate_magnitude(self.calculate_vector(thumb_tip, index_tip))
        
        # Pinch is detected when fingertips are close but not touching
        # Adjust threshold as needed for sensitivity
        return distance < 30  # Pixels
    
    def process_gestures(self, landmarks):
        """
        Process all possible gestures and return detected ones with improved stability.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            Dictionary of detected gestures and their states
        """
        # Update landmark history
        self.update_landmarks(landmarks)
        
        # Initialize results
        detected_gestures = {}
        current_time = time.time()
        
        # Keep track of active gestures from the previous frame
        persistent_gestures = getattr(self, 'persistent_gestures', {})
        self.active_gestures = {}
        
        # Required frames to deactivate (more than to activate for stability)
        deactivation_threshold = self.min_consecutive_detections * 2
        
        # Detect swipe gestures
        swipe_gesture = self.detect_swipe()
        if swipe_gesture:
            # Update state of specific swipe direction
            self.detection_counters[swipe_gesture] += 1
            # Reset counter for the opposite swipe
            opposite_swipe = 'swipe_right' if swipe_gesture == 'swipe_left' else 'swipe_left'
            self.detection_counters[opposite_swipe] = 0
            
            # Check if we've seen this gesture enough consecutive times
            if self.detection_counters[swipe_gesture] >= self.min_consecutive_detections:
                # Check cooldown period
                if (current_time - self.last_gesture_time[swipe_gesture] 
                    > self.gesture_cooldown):
                    # Confirm the gesture
                    self.gesture_states[swipe_gesture] = self.STATES['CONFIRMED']
                    self.last_gesture_time[swipe_gesture] = current_time
                    detected_gestures[swipe_gesture] = True
                    self.active_gestures[swipe_gesture] = True
                    # Add to persistent gestures with a counter
                    persistent_gestures[swipe_gesture] = deactivation_threshold
                else:
                    # In cooldown period
                    self.gesture_states[swipe_gesture] = self.STATES['COOLDOWN']
        else:
            # If the gesture was active in previous frames, decrease its counter
            for swipe in ['swipe_left', 'swipe_right']:
                if swipe in persistent_gestures:
                    persistent_gestures[swipe] -= 1
                    if persistent_gestures[swipe] > 0:
                        # Still consider it active
                        detected_gestures[swipe] = True
                        self.active_gestures[swipe] = True
                    else:
                        # Reset completely
                        self.detection_counters[swipe] = 0
                        persistent_gestures.pop(swipe, None)
        
        # Detect thumbs gestures
        thumbs_gesture = self.detect_thumbs_gesture(landmarks)
        if thumbs_gesture:
            # Update detection counter
            self.detection_counters[thumbs_gesture] += 1
            # Reset counter for the opposite thumbs gesture
            opposite_thumbs = 'thumbs_down' if thumbs_gesture == 'thumbs_up' else 'thumbs_up'
            self.detection_counters[opposite_thumbs] = 0
            
            if self.detection_counters[thumbs_gesture] >= self.min_consecutive_detections:
                # Check cooldown period
                if (current_time - self.last_gesture_time[thumbs_gesture] 
                    > self.gesture_cooldown):
                    # Confirm the gesture
                    self.gesture_states[thumbs_gesture] = self.STATES['CONFIRMED']
                    self.last_gesture_time[thumbs_gesture] = current_time
                    detected_gestures[thumbs_gesture] = True
                    self.active_gestures[thumbs_gesture] = True
                    # Add to persistent gestures with a counter
                    persistent_gestures[thumbs_gesture] = deactivation_threshold
                else:
                    # In cooldown period
                    self.gesture_states[thumbs_gesture] = self.STATES['COOLDOWN']
        else:
            # If the gesture was active in previous frames, decrease its counter
            for thumbs in ['thumbs_up', 'thumbs_down']:
                if thumbs in persistent_gestures:
                    persistent_gestures[thumbs] -= 1
                    if persistent_gestures[thumbs] > 0:
                        # Still consider it active
                        detected_gestures[thumbs] = True
                        self.active_gestures[thumbs] = True
                    else:
                        # Reset completely
                        self.detection_counters[thumbs] = 0
                        persistent_gestures.pop(thumbs, None)
        
        # Detect index pointing (for laser pointer control)
        is_pointing = self.detect_index_pointing(landmarks)
        if is_pointing:
            detected_gestures['index_pointing'] = True
            self.active_gestures['index_pointing'] = True
            persistent_gestures['index_pointing'] = deactivation_threshold
        else:
            # If the gesture was active in previous frames, decrease its counter
            if 'index_pointing' in persistent_gestures:
                persistent_gestures['index_pointing'] -= 1
                if persistent_gestures['index_pointing'] > 0:
                    # Still consider it active
                    detected_gestures['index_pointing'] = True
                    self.active_gestures['index_pointing'] = True
                else:
                    # Reset completely
                    self.detection_counters['index_pointing'] = 0
                    persistent_gestures.pop('index_pointing', None)
        
        # Detect pinch gesture for annotation
        is_pinching = self.detect_pinch(landmarks)
        if is_pinching:
            detected_gestures['pinch'] = True
            self.active_gestures['pinch'] = True
            persistent_gestures['pinch'] = deactivation_threshold
        else:
            # If the gesture was active in previous frames, decrease its counter
            if 'pinch' in persistent_gestures:
                persistent_gestures['pinch'] -= 1
                if persistent_gestures['pinch'] > 0:
                    # Still consider it active
                    detected_gestures['pinch'] = True
                    self.active_gestures['pinch'] = True
                else:
                    # Reset completely
                    self.detection_counters['pinch'] = 0
                    persistent_gestures.pop('pinch', None)
        
        # Update persistent gestures
        self.persistent_gestures = persistent_gestures
        
        return detected_gestures


class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initialize the hand detector with MediaPipe.
        """
        print("Initializing MediaPipe Hand Detection...")
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create Hands object with specified parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Landmark history for tracking
        self.landmark_history = []
        self.history_size = 10
        
        print("MediaPipe Hand Detection initialized successfully")
    
    def find_hands(self, frame, draw=True):
        """
        Detect hand landmarks in a frame.
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        self.results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for display
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Initialize empty landmarks list
        landmarks = []
        hand_present = False
        
        # Check if hands were detected
        if self.results.multi_hand_landmarks:
            hand_present = True
            
            # Process first hand only (as per max_hands=1)
            hand_landmarks = self.results.multi_hand_landmarks[0]
            
            # Extract landmark coordinates
            h, w, c = frame.shape
            for lm in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                px, py, pz = int(lm.x * w), int(lm.y * h), lm.z
                landmarks.append((px, py, pz))
            
            # Draw landmarks on frame if requested
            if draw:
                self.mp_drawing.draw_landmarks(
                    processed_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
            # Add landmarks to history
            self.update_landmark_history(landmarks)
        
        return processed_frame, landmarks, hand_present
    
    def update_landmark_history(self, landmarks):
        """Update the landmark history with new landmarks."""
        if landmarks:
            self.landmark_history.append(landmarks)
            
            # Keep history at desired size
            if len(self.landmark_history) > self.history_size:
                self.landmark_history.pop(0)
    
    def get_landmark_history(self):
        """Get the history of hand landmarks."""
        return self.landmark_history
    
    def get_palm_center(self, landmarks=None):
        """Calculate the center of the palm from landmarks."""
        if landmarks is None:
            if not self.landmark_history:
                return None
            landmarks = self.landmark_history[-1]
        
        # Use wrist (0) and middle finger MCP (9) to estimate palm center
        if len(landmarks) >= 21:
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            
            palm_x = (wrist[0] + middle_mcp[0]) // 2
            palm_y = (wrist[1] + middle_mcp[1]) // 2
            
            return (palm_x, palm_y)
        
        return None
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


class VideoProcessor:
    def __init__(self, camera_id=0, width=640, height=480, fps=60, backend=None):
        """
        Initialize the video processor with camera settings.
        
        Args:
            camera_id: Camera device ID
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            backend: OpenCV backend to use (if None, use default)
        """
        print(f"Initializing camera with ID: {camera_id}")
        
        # Open camera with specified backend if provided
        if backend is not None:
            self.cap = cv2.VideoCapture(camera_id, backend)
        else:
            # Try DirectShow API on Windows
            try:
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            except:
                # Fallback to default API
                self.cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties - try to get higher FPS for smoother tracking
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Wait for camera to initialize
        time.sleep(1)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open camera {camera_id}")
        
        # Take a test frame
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise ValueError(f"Error: Could not read frame from camera {camera_id}")
        
        print(f"Camera initialized successfully")
    
    def preprocess_frame(self, frame):
        """
        Apply basic preprocessing to reduce noise and normalize frame.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            processed: Processed frame with noise reduction
        """
        # Apply Gaussian blur for noise reduction
        processed = cv2.GaussianBlur(frame, (5, 5), 0)
        return processed
    
    def adapt_to_lighting(self, frame):
       """
       Adapt frame processing based on current lighting conditions.
       
       Args:
           frame: Input frame from webcam
           
       Returns:
           adapted_frame: Frame with adaptive processing applied
       """
       # Convert to HSV color space
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
       # Calculate average brightness (V channel)
       _, _, v = cv2.split(hsv)
       avg_brightness = np.mean(v)
       
       # Apply adaptive processing based on brightness
       if avg_brightness < 50:  # Low light
           # Increase brightness
           hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=1.5, beta=10)
           adapted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
       elif avg_brightness > 200:  # Too bright
           # Decrease brightness
           hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=0.8, beta=0)
           adapted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
       else:
           # Normal lighting
           adapted_frame = frame
       
       return adapted_frame
   
    def read_frame(self):
       """
       Read and preprocess a frame from the webcam.
       
       Returns:
           success: Boolean indicating if frame was read successfully
           original_frame: Original unprocessed frame
           processed_frame: Frame after preprocessing
       """
       success, original_frame = self.cap.read()
       
       if not success:
           return False, None, None
       
       # Apply lighting adaptation
       lighting_adapted = self.adapt_to_lighting(original_frame)
       
       # Apply noise reduction
       processed_frame = self.preprocess_frame(lighting_adapted)
       
       return success, original_frame, processed_frame
   
    def release(self):
       """Release the video capture resource."""
       if self.cap.isOpened():
           self.cap.release()
           print("Camera released.")


def find_working_camera(silent=True):
    """
    Find the first working camera on the system by trying different backends.
    
    Args:
        silent: Whether to suppress verbose output
        
    Returns:
        camera_id: ID of the first working camera, or None if none found
        backend: The backend that worked, or None
    """
    if not silent:
        print("Searching for available cameras...")
    
    # DirectShow works well on Windows, try it first
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                if not silent:
                    print(f"Found working camera with ID: 0 using DirectShow backend")
                return 0, cv2.CAP_DSHOW
            cap.release()
    except Exception:
        pass
    
    # Try other backends only if DirectShow fails
    backends = [cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_V4L2]
    
    for camera_id in range(2):  # Just try the first two cameras
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.release()
                        if not silent:
                            print(f"Found working camera with ID: {camera_id}")
                        return camera_id, backend
                    cap.release()
            except Exception:
                continue
    
    # Try a last resort approach - just camera 0 with no specified backend
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                if not silent:
                    print(f"Found working camera with basic access")
                return 0, None
            cap.release()
    except Exception:
        pass
    
    if not silent:
        print("No working cameras found.")
    return None, None


class LaserPointerController:
    """Controls the virtual laser pointer functionality."""
    
    def __init__(self, laser_color=(255, 0, 0), laser_size=8, trail_length=5):
        """
        Initialize the laser pointer controller with customizable settings.
        
        Args:
            laser_color: RGB color tuple for the laser pointer (default: red)
            laser_size: Size of the laser point in pixels
            trail_length: Length of the trailing effect
        """
        # Store settings
        self.laser_color = laser_color
        self.laser_size = laser_size
        self.trail_length = trail_length
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Track position history for trail effect
        self.position_history = collections.deque(maxlen=trail_length)
        
        # Camera dimensions for mapping
        self.camera_width = 640  # Default
        self.camera_height = 480  # Default
        self.horizontal_flip = True  # Mirror horizontally for natural movement
        
        # Smoothing factor for more natural movement
        self.smoothing_factor = 0.4  # Higher value = less smoothing
        
        # Last position for smoothing
        self.last_position = None
        
        # Current position on screen
        self.current_position = None
        
        # Mouse state
        self.is_mouse_down = False
        
        # Edge padding to prevent edge issues
        self.edge_padding = 50  # Pixels
        
        # Create the overlay window for presentation mode
        self.overlay = LaserOverlay(laser_color=laser_color, 
                                   laser_size=laser_size, 
                                   trail_length=trail_length)
        
        # Presentation mode flag
        self.presentation_mode = False
        
        # Annotation state
        self.annotation_active = False
        
        print(f"Laser Pointer initialized with color: RGB{laser_color}")
    
    def set_camera_dimensions(self, width, height):
        """
        Set camera dimensions for coordinate mapping.
        
        Args:
            width: Camera frame width in pixels
            height: Camera frame height in pixels
        """
        self.camera_width = width
        self.camera_height = height
        print(f"Laser Pointer: Camera dimensions set to {width}x{height}")
    
    def update_position(self, index_tip_position):
        """
        Update the laser pointer position based on index fingertip.
        
        Args:
            index_tip_position: (x, y) position of index fingertip in camera space
            
        Returns:
            (x, y) position of laser pointer in camera space
        """
        if index_tip_position is None:
            return None
        
        # Apply horizontal flipping if needed
        if self.horizontal_flip:
            camera_x = self.camera_width - index_tip_position[0]
            camera_y = index_tip_position[1]
        else:
            camera_x, camera_y = index_tip_position
        
        # Apply smoothing if we have a previous position
        if self.last_position:
            smoothed_x = int(self.last_position[0] * (1 - self.smoothing_factor) + 
                          camera_x * self.smoothing_factor)
            smoothed_y = int(self.last_position[1] * (1 - self.smoothing_factor) + 
                          camera_y * self.smoothing_factor)
            
            current_position = (smoothed_x, smoothed_y)
        else:
            current_position = (camera_x, camera_y)
        
        # Update last position for next frame
        self.last_position = current_position
        
        # Add to position history for trail effect
        self.position_history.append(current_position)
        
        return current_position
    def map_to_screen(self, camera_position):
        """
        Map camera coordinates to screen coordinates.
        
        Args:
            camera_position: (x, y) position in camera space
            
        Returns:
            (x, y) position mapped to screen coordinates
        """
        if camera_position is None:
            return None
        
        # Apply edge padding to prevent edge issues
        camera_x, camera_y = camera_position
        
        # Constrain within padding
        camera_x = max(self.edge_padding, min(camera_x, self.camera_width - self.edge_padding))
        
        # Map to screen coordinates with padding adjustment
        x_ratio = (camera_x - self.edge_padding) / (self.camera_width - 2 * self.edge_padding)
        y_ratio = camera_y / self.camera_height
        
        # Calculate screen position
        screen_x = int(x_ratio * self.screen_width)
        screen_y = int(y_ratio * self.screen_height)
        
        # Save current screen position
        self.current_position = (screen_x, screen_y)
        
        return (screen_x, screen_y)
    
    def move_cursor(self, camera_position):
        """
        Move the system cursor to match the laser pointer position.
        
        Args:
            camera_position: (x, y) position in camera coordinates
            
        Returns:
            (x, y) position where cursor was moved on screen
        """
        if camera_position is None:
            return None
        
        # Map to screen coordinates
        screen_position = self.map_to_screen(camera_position)
        
        if screen_position:
            try:
                # Update the overlay if in presentation mode
                if self.presentation_mode:
                    self.overlay.update_overlay(screen_position)
                
                # Move the actual system cursor with best available method
                if HAVE_WIN32API:
                    # Direct Win32 API is much more responsive
                    win32api.SetCursorPos(screen_position)
                else:
                    # PyAutoGUI fallback
                    pyautogui.moveTo(screen_position[0], screen_position[1], _pause=False)
            except Exception as e:
                logger.error(f"Error moving cursor: {e}")
        
        return screen_position
    
    def set_presentation_mode(self, active):
        """
        Set presentation mode status to control overlay visibility.
        
        Args:
            active: Boolean indicating if presentation is active
        """
        self.presentation_mode = active
        
        # Show/hide overlay based on presentation mode
        if active:
            # Show overlay and hide cursor
            self.overlay.show()
            if HAVE_WIN32API:
                win32api.ShowCursor(False)
        else:
            # Hide overlay and show cursor
            self.overlay.hide()
            if HAVE_WIN32API:
                win32api.ShowCursor(True)
    
    def toggle_annotation(self):
        """Toggle annotation mode on/off."""
        self.annotation_active = not self.annotation_active
        
        if self.annotation_active:
            # Start a new annotation
            self.overlay.start_annotation()
            print("Annotation mode activated")
        else:
            # Stop and save the current annotation
            self.overlay.stop_annotation()
            print("Annotation mode deactivated")
        
        return self.annotation_active
    
    def clear_annotations(self):
        """Clear all annotations."""
        self.overlay.clear_annotations()
        return True
    
    def click(self):
        """Perform a mouse click at current laser position."""
        try:
            if HAVE_WIN32API:
                # Use Win32 API for more reliable clicks
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.05)  # Small delay for better reliability
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            else:
                # PyAutoGUI fallback
                pyautogui.click(_pause=False)
            
            logger.info(f"Laser click at {self.current_position}")
        except Exception as e:
            logger.error(f"Error performing click: {e}")
            
    def mouse_down(self):
        """Press and hold mouse button at current laser position."""
        if not self.is_mouse_down:
            try:
                if HAVE_WIN32API:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                else:
                    pyautogui.mouseDown(_pause=False)
                
                self.is_mouse_down = True
                logger.info(f"Mouse down at {self.current_position}")
            except Exception as e:
                logger.error(f"Error performing mouse down: {e}")
    
    def mouse_up(self):
        """Release mouse button."""
        if self.is_mouse_down:
            try:
                if HAVE_WIN32API:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                else:
                    pyautogui.mouseUp(_pause=False)
                
                self.is_mouse_down = False
                logger.info(f"Mouse up at {self.current_position}")
            except Exception as e:
                logger.error(f"Error performing mouse up: {e}")
    
    def draw_laser(self, frame, position, active=True):
        """
        Draw the laser pointer on the frame.
        
        Args:
            frame: The frame to draw on
            position: (x, y) position in camera coordinates
            active: Whether the laser is active
            
        Returns:
            The frame with laser pointer drawn
        """
        if position is None or not active:
            return frame
        
        # Make a copy of the frame to avoid modifying the original
        result_frame = frame.copy()
        
        # Draw trailing effect first (fading gradient)
        positions = list(self.position_history)
        if len(positions) >= 2:
            # Draw lines connecting the positions with decreasing opacity
            for i in range(1, len(positions)):
                # Calculate opacity based on position in history (older = more transparent)
                opacity = 0.7 * (i / len(positions))
                
                # Get points for this segment
                p1 = positions[i-1]
                p2 = positions[i]
                
                # Create gradient color with decreasing opacity
                color = self.laser_color
                
                # Draw line segment with appropriate thickness and color
                cv2.line(result_frame, p1, p2, color, 
                        thickness=max(1, int(self.laser_size * 0.3 * opacity)),
                        lineType=cv2.LINE_AA)
        
        # Draw main laser dot at current position
        cv2.circle(result_frame, position, self.laser_size, self.laser_color, -1, lineType=cv2.LINE_AA)
        
        # Add a highlight effect inside the dot for better visibility
        highlight_size = max(1, int(self.laser_size * 0.5))
        highlight_offset = max(1, int(self.laser_size * 0.3))
        highlight_pos = (position[0] - highlight_offset, position[1] - highlight_offset)
        cv2.circle(result_frame, highlight_pos, highlight_size, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        
        return result_frame
    
    def reset(self):
        """Reset laser pointer state."""
        # Ensure mouse button is released
        self.mouse_up()
        
        # Clear history
        self.position_history.clear()
        self.last_position = None
        
        # Cleanup overlay
        self.overlay.cleanup()


class CommandInterface:
    """Interface for translating gestures to system commands."""
    
    def __init__(self):
        """Initialize the command interface."""
        # Command throttling
        self.last_command_time = {}
        self.command_cooldown = 0.8  # Reduced from 1.0 for better responsiveness
        
        # Command queue for asynchronous execution
        self.command_queue = queue.Queue()
        
        # Command history for debugging and state tracking
        self.command_history = deque(maxlen=10)
        
        # Active presentation state
        self.presentation_active = False
        
        # Start command processing thread
        self.exit_flag = False
        self.command_thread = threading.Thread(target=self._command_processor)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        # Visual feedback state
        self.current_feedback = None
        self.feedback_time = 0
        self.feedback_duration = 1.5  # seconds to show feedback
        
        # Define gesture to command mappings
        self.gesture_commands = {
            'swipe_left': {
                'key': 'right',
                'description': 'Next Slide',
                'requires_presentation': True
            },
            'swipe_right': {
                'key': 'left',
                'description': 'Previous Slide',
                'requires_presentation': True
            },
            'thumbs_up': {
                'key': 'f5',
                'description': 'Start Presentation',
                'requires_presentation': False,
                'action': self._start_presentation
            },
            'thumbs_down': {
                'key': 'escape',
                'description': 'End Presentation',
                'requires_presentation': True,
                'action': self._end_presentation
            },
            'index_pointing': {
                'description': 'Laser Pointer',
                'requires_presentation': False
            },
            'pinch': {
                'description': 'Toggle Annotation',
                'requires_presentation': True
            }
        }
        
        logger.info("Command Interface initialized")
    
    def process_gestures(self, detected_gestures, landmarks=None):
        """
        Process detected gestures and convert to commands.
        
        Args:
            detected_gestures: Dictionary of detected gestures
            landmarks: Hand landmarks for pointer positioning (optional)
            
        Returns:
            feedback: Visual feedback information for the UI
        """
        current_time = time.time()
        commands_to_execute = []
        feedback = None
        
        # Process each detected gesture
        for gesture, active in detected_gestures.items():
            if not active:
                continue
                
            # Check if gesture is defined in our mappings
            if gesture in self.gesture_commands:
                command_info = self.gesture_commands[gesture]
                
                # Check if presentation state is appropriate for this command
                if command_info.get('requires_presentation', False) and not self.presentation_active:
                    # Skip commands that require active presentation
                    continue
                
                # Skip laser pointer gesture (handled separately)
                if gesture == 'index_pointing':
                    continue
                
                # Check cooldown period
                last_time = self.last_command_time.get(gesture, 0)
                if current_time - last_time < self.command_cooldown:
                    # Command is still in cooldown
                    continue
                
                # Update command time
                self.last_command_time[gesture] = current_time
                
                # Create feedback
                feedback = {
                    'gesture': gesture,
                    'command': command_info.get('description', gesture),
                    'time': current_time
                }
                
                # If command has a special action function
                if 'action' in command_info and command_info['action'] is not None:
                    # Queue special action
                    self.command_queue.put({
                        'type': 'action',
                        'action': command_info['action'],
                        'gesture': gesture,
                        'time': current_time
                    })
                # If command has a key to press
                elif 'key' in command_info:
                    # Queue key press
                    self.command_queue.put({
                        'type': 'key',
                        'key': command_info['key'],
                        'gesture': gesture,
                        'time': current_time
                    })
                
                logger.info(f"Queued command: {gesture} -> {command_info.get('description', gesture)}")
        
        # Return feedback for UI
        if feedback:
            self.current_feedback = feedback
            self.feedback_time = current_time
            
        return self.current_feedback
    
    def get_feedback(self):
        """
        Get current visual feedback if it's still active.
        
        Returns:
            Current feedback info or None if expired
        """
        current_time = time.time()
        if self.current_feedback and current_time - self.feedback_time < self.feedback_duration:
            return self.current_feedback
        else:
            self.current_feedback = None
            return None
    
    def _command_processor(self):
        """Background thread for processing commands from the queue."""
        while not self.exit_flag:
            try:
                # Get command with a timeout to allow checking exit_flag
                try:
                    command = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process based on command type
                if command['type'] == 'key':
                    # Press the specified key
                    try:
                        pyautogui.press(command['key'])
                        logger.info(f"Executed key press: {command['key']}")
                    except Exception as e:
                        logger.error(f"Error executing key press {command['key']}: {str(e)}")
                
                elif command['type'] == 'action':
                    # Execute the action function
                    try:
                        command['action']()
                        logger.info(f"Executed action for gesture: {command['gesture']}")
                    except Exception as e:
                        logger.error(f"Error executing action for {command['gesture']}: {str(e)}")
                
                # Add to history
                self.command_history.append({
                    'command': command,
                    'time': time.time()
                })
                
                # Mark as done
                self.command_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in command processor: {str(e)}")
    
    def _start_presentation(self):
        """Start presentation mode."""
        self.presentation_active = True
        logger.info("Presentation started")
    
    def _end_presentation(self):
        """End presentation mode."""
        self.presentation_active = False
        logger.info("Presentation ended")
    
    def shutdown(self):
        """Clean up resources and stop threads."""
        self.exit_flag = True
        if self.command_thread.is_alive():
            self.command_thread.join(timeout=1.0)
        logger.info("Command Interface shut down")


class PowerPointController:
    """Controls PowerPoint presentations programmatically."""
    
    def __init__(self, laser_controller=None):
        """
        Initialize PowerPoint controller.
        
        Args:
            laser_controller: Optional LaserPointerController instance
        """
        self.presentation_active = False
        self.ppt_app = None
        self.win32com_available = False
        
        # Store laser controller reference
        self.laser_controller = laser_controller
        
        # Try to import the win32com module
        try:
            import win32com.client
            self.win32com_available = True
            logger.info("win32com.client is available for direct PowerPoint control")
        except ImportError:
            logger.info("win32com.client is not available, will use keyboard shortcuts instead")
    
    def connect_to_powerpoint(self):
        """Connect to running PowerPoint application."""
        if not self.win32com_available:
            logger.info("Direct PowerPoint control not available, using keyboard shortcuts instead")
            return False
            
        try:
            import win32com.client
            self.ppt_app = win32com.client.Dispatch("PowerPoint.Application")
            logger.info("Connected to PowerPoint application")
            return True
        except Exception as e:
            logger.error(f"Could not connect to PowerPoint: {str(e)}")
            return False
    
    def start_presentation(self):
        """Start the active presentation."""
        if not self.win32com_available:
            # Use keyboard shortcut (F5) instead
            pyautogui.press('f5')
            self.presentation_active = True
            
            # Enable laser pointer overlay
            if self.laser_controller:
                self.laser_controller.set_presentation_mode(True)
                
            logger.info("Started presentation using keyboard shortcut")
            return True
            
        try:
            if self.ppt_app:
                # Get the active presentation
                presentation = self.ppt_app.ActivePresentation
                # Start from beginning
                presentation.SlideShowSettings.Run()
                self.presentation_active = True
                
                # Enable laser pointer overlay
                if self.laser_controller:
                    self.laser_controller.set_presentation_mode(True)
                    
                logger.info("PowerPoint presentation started via COM")
                return True
        except Exception as e:
            logger.error(f"Error starting presentation: {str(e)}")
            # Fall back to keyboard shortcut
            pyautogui.press('f5')
            self.presentation_active = True
            
            # Enable laser pointer overlay
            if self.laser_controller:
                self.laser_controller.set_presentation_mode(True)
                
            logger.info("Fell back to keyboard shortcut to start presentation")
        return True
    
    def next_slide(self):
        """Move to the next slide."""
        if not self.win32com_available or not self.ppt_app:
            # Use keyboard shortcut (right arrow) instead
            pyautogui.press('right')
            logger.info("Next slide using keyboard shortcut")
            return True
            
        try:
            if self.ppt_app and self.presentation_active:
                self.ppt_app.SlideShowWindows(1).View.Next()
                logger.info("Next slide via COM")
                return True
        except Exception as e:
            logger.error(f"Error moving to next slide: {str(e)}")
            # Fall back to keyboard shortcut
            pyautogui.press('right')
            logger.info("Fell back to keyboard shortcut for next slide")
        return True
    
    def previous_slide(self):
        """Move to the previous slide."""
        if not self.win32com_available or not self.ppt_app:
            # Use keyboard shortcut (left arrow) instead
            pyautogui.press('left')
            logger.info("Previous slide using keyboard shortcut")
            return True
            
        try:
            if self.ppt_app and self.presentation_active:
                self.ppt_app.SlideShowWindows(1).View.Previous()
                logger.info("Previous slide via COM")
                return True
        except Exception as e:
            logger.error(f"Error moving to previous slide: {str(e)}")
            # Fall back to keyboard shortcut
            pyautogui.press('left')
            logger.info("Fell back to keyboard shortcut for previous slide")
        return True                
    
    def end_presentation(self):
        """End the presentation."""
        if not self.win32com_available or not self.ppt_app:
            # Use keyboard shortcut (Escape) instead
            pyautogui.press('escape')
            self.presentation_active = False
            
            # Disable laser pointer overlay
            if self.laser_controller:
                self.laser_controller.set_presentation_mode(False)
                
            logger.info("Ended presentation using keyboard shortcut")
            return True
            
        try:
            if self.ppt_app and self.presentation_active:
                self.ppt_app.SlideShowWindows(1).View.Exit()
                self.presentation_active = False
                
                # Disable laser pointer overlay
                if self.laser_controller:
                    self.laser_controller.set_presentation_mode(False)
                    
                logger.info("PowerPoint presentation ended via COM")
                return True
        except Exception as e:
            logger.error(f"Error ending presentation: {str(e)}")
            # Fall back to keyboard shortcut
            pyautogui.press('escape')
            self.presentation_active = False
            
            # Disable laser pointer overlay
            if self.laser_controller:
                self.laser_controller.set_presentation_mode(False)
                
            logger.info("Fell back to keyboard shortcut to end presentation")
        return True
    
    def is_presentation_running(self):
        """Check if a presentation is currently running."""
        if not self.win32com_available or not self.ppt_app:
            return self.presentation_active  # Use the tracked state
            
        try:
            if self.ppt_app:
                is_running = self.ppt_app.SlideShowWindows.Count > 0
                
                # Update presentation state if changed
                if is_running != self.presentation_active:
                    self.presentation_active = is_running
                    
                    # Update laser pointer overlay state
                    if self.laser_controller:
                        self.laser_controller.set_presentation_mode(is_running)
                
                return is_running
        except:
            pass
        return self.presentation_active
    
    def shutdown(self):
        """Release COM resources."""
        if self.win32com_available and self.ppt_app:
            try:
                # Don't close PowerPoint, just release the COM object
                self.ppt_app = None
                logger.info("Released PowerPoint COM object")
            except:
                pass


def draw_command_feedback(frame, feedback):
    """
    Draw command feedback on the frame.
    
    Args:
        frame: The frame to draw on
        feedback: Feedback information from command interface
    """
    if not feedback:
        return frame
    
    # Extract feedback info
    gesture = feedback.get('gesture', '')
    command = feedback.get('command', '')
    time_elapsed = time.time() - feedback.get('time', 0)
    
    # Only show if within duration
    if time_elapsed > 1.5:  # 1.5 seconds display time
        return frame
    
    # Calculate fade-out effect (1.0 to 0.0 over duration)
    alpha = max(0, 1.0 - (time_elapsed / 1.5))
    
    # Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Background for contrast
    bg_color = (0, 0, 0)
    text_color = (0, 255, 0)  # Green by default
    
    # Adjust color based on gesture type
    if 'swipe' in gesture:
        text_color = (0, 255, 255)  # Yellow
    elif 'thumbs_up' in gesture:
        text_color = (0, 255, 0)  # Green
    elif 'thumbs_down' in gesture:
        text_color = (0, 0, 255)  # Red
    elif 'pinch' in gesture:
        text_color = (255, 165, 0)  # Orange
    
    # Apply alpha fade to text color
    text_color = tuple(int(c * alpha) for c in text_color)
    
    # Format text
    text = f"Command: {command}"
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position at bottom center
    x = (frame.shape[1] - text_width) // 2
    y = frame.shape[0] - 50
    
    # Make sure we're within frame boundaries
    if (y-text_height-10 < 0 or y+10 >= frame.shape[0] or 
        x-10 < 0 or x+text_width+10 >= frame.shape[1]):
        # Just draw text without background if we're at the edge
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
        return frame
    
    # Draw semi-transparent background - safe way
    try:
        # Get the region of interest
        sub_img = frame[y-text_height-10:y+10, x-10:x+text_width+10].copy()
        
        # Create a black rectangle with the same data type as the sub image
        black_rect = np.zeros_like(sub_img)
        
        # Create the blended image manually to avoid type issues
        blended = cv2.addWeighted(sub_img, 1-alpha*0.8, black_rect, alpha*0.8, 0)
        
        # Put the blended region back
        frame[y-text_height-10:y+10, x-10:x+text_width+10] = blended
    except Exception as e:
        print(f"Warning: Could not draw background: {e}")
    
    # Draw text with specified settings
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
    
    return frame


def draw_presentation_status(frame, presentation_active):
    """Draw the current presentation status on the frame."""
    status = "PRESENTATION ACTIVE" if presentation_active else "PRESENTATION INACTIVE"
    color = (0, 255, 0) if presentation_active else (0, 0, 255)
    
    cv2.putText(frame, status, (10, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame


def draw_gesture_command_info(frame, detected_gestures, command_interface):
    """
    Draw detected gesture information with command mapping on the frame.
    
    Args:
        frame: The frame to draw on
        detected_gestures: Dictionary of detected gestures
        command_interface: The command interface instance
    """
    # Position and style settings
    y_pos = 70
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    bg_color = (50, 50, 50)
    
    # Draw background for text
    cv2.rectangle(frame, (10, y_pos-30), (320, y_pos+150), bg_color, -1)
    
    # Draw title
    cv2.putText(frame, "Gesture Commands:", (15, y_pos), 
               font, font_scale+0.1, (255, 255, 255), thickness+1)
    
    # Draw gesture statuses
    y_pos += 30
    
    # List of all possible gestures to display
    all_gestures = [
        'swipe_left', 'swipe_right', 'thumbs_up', 'thumbs_down', 'index_pointing', 'pinch'
    ]
    
    for gesture in all_gestures:
        active = gesture in detected_gestures and detected_gestures[gesture]
        status = "ACTIVE" if active else "INACTIVE"
        color = (0, 255, 0) if active else (100, 100, 100)
        
        # Get the command description from the command interface
        command_desc = ""
        if gesture in command_interface.gesture_commands:
            command_info = command_interface.gesture_commands[gesture]
            command_desc = f" → {command_info.get('description', '')}"
        
        # Make the text more user-friendly
        display_text = gesture.replace('_', ' ').title()
        cv2.putText(frame, f"{display_text}{command_desc}", (15, y_pos), 
                   font, font_scale, color, thickness)
        y_pos += 25
    
    return frame


def terminal_input_thread():
    """Thread function to monitor terminal input for exit command."""
    global exit_flag
    print("Type 'exit' in the terminal to stop the program:")
    
    while not exit_flag:
        cmd = input().strip().lower()
        if cmd == 'exit':
            print("Exit command received. Stopping...")
            exit_flag = True
            break
        elif cmd in ['next', 'previous', 'start', 'end']:
            print(f"Command received: {cmd}")
            # You can add handling for these commands if needed


def control_powerpoint_with_laser():
    """Main function for PowerPoint control with laser pointer."""
    global exit_flag
    
    # First find a working camera
    camera_id, backend = find_working_camera(silent=False)
    
    if camera_id is None:
        print("Error: No working camera found. Please check your camera connection.")
        return
    
    # Start the terminal input monitoring thread
    input_thread = threading.Thread(target=terminal_input_thread)
    input_thread.daemon = True  # Thread will exit when main program exits
    input_thread.start()
    
    try:
        # Initialize video processor with the working camera
        # Try for higher framerate (60 FPS) for smoother tracking if camera supports it
        if backend is not None:
            processor = VideoProcessor(camera_id=camera_id, backend=backend, fps=60)
        else:
            processor = VideoProcessor(camera_id=camera_id, fps=60)
        
        # Initialize hand detector with higher confidence for more stable detection
        detector = HandDetector(
            static_mode=False, 
            max_hands=1, 
            detection_confidence=0.7, 
            tracking_confidence=0.7  # Increased tracking confidence
        )
        
        # Initialize gesture recognizer
        gesture_recognizer = GestureRecognizer(history_size=5)  # Reduced history size
        
        # Initialize command interface with reduced cooldown
        command_interface = CommandInterface()
        command_interface.command_cooldown = 0.8  # Slightly faster response
        
        # Initialize laser pointer controller (RED laser by default)
        laser_controller = LaserPointerController(laser_color=(255, 0, 0), laser_size=8, trail_length=5)
        
        # Get camera dimensions
        success, original, processed = processor.read_frame()
        if success:
            frame_height, frame_width = processed.shape[:2]
            laser_controller.set_camera_dimensions(frame_width, frame_height)
        
        # Initialize PowerPoint controller with laser controller reference
        ppt_controller = PowerPointController(laser_controller=laser_controller)
        ppt_controller.connect_to_powerpoint()
        
        # Override gesture command mappings for PowerPoint
        command_interface.gesture_commands.update({
            'swipe_left': {
                'description': 'Next Slide',
                'requires_presentation': True,
                'action': lambda: ppt_controller.next_slide()
            },
            'swipe_right': {
                'description': 'Previous Slide',
                'requires_presentation': True,
                'action': lambda: ppt_controller.previous_slide()
            },
            'thumbs_up': {
                'description': 'Start Presentation',
                'requires_presentation': False,
                'action': lambda: ppt_controller.start_presentation() 
            },
            'thumbs_down': {
                'description': 'End Presentation',
                'requires_presentation': True,
                'action': lambda: ppt_controller.end_presentation()
            },
            'index_pointing': {
                'description': 'Laser Pointer',
                'requires_presentation': False
            },
            'pinch': {
                'description': 'Toggle Annotation',
                'requires_presentation': True,
                'action': lambda: laser_controller.toggle_annotation()
            }
        })
        
        print("Starting PowerPoint Presentation Controller with Laser Pointer...")
        print("Type 'exit' in the terminal to stop the program")
        print("\nGesture Commands:")
        print("  • Swipe Left → Next Slide")
        print("  • Swipe Right → Previous Slide")
        print("  • Thumbs Up → Start Presentation")
        print("  • Thumbs Down → End Presentation")
        print("  • Index Finger Pointing → Virtual Laser Pointer")
        print("  • Index-Thumb Pinch → Toggle Annotation Mode")
        print("\nOpen your PowerPoint presentation and use Thumbs Up to start it")
        print("Press 'l' to cycle through different laser colors (Red, Green, Blue)")
        print("Press 'a' to toggle annotation mode, 'c' to clear annotations")
        
        # Track presentation status
        presentation_active = False
        
        # FIX: Use higher process priority if on Windows
        if HAVE_WIN32API:
            try:
                import psutil
                p = psutil.Process(os.getpid())
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                print("Running with high process priority for better performance")
            except:
                pass
        
        # Main processing loop
        frame_count = 0
        while not exit_flag:
            # Read and process frame
            success, original, processed = processor.read_frame()
            
            if not success:
                print("Error: Failed to read frame. Retrying...")
                continue
            
            # Only process every other frame if system is struggling
            frame_count += 1
            process_frame = True
            
            # Detect hands in processed frame
            hand_frame, landmarks, hand_present = detector.find_hands(processed, draw=process_frame)
            
            # Dictionary for this frame's active gestures
            detected_gestures = {}
            
            # Track laser position for visualization
            laser_position = None
            laser_active = False
            
            # Process gestures if hand is present
            if hand_present and landmarks and process_frame:
                # Detect gestures
                detected_gestures = gesture_recognizer.process_gestures(landmarks)
                
                # Handle pinch gesture for annotation
                if 'pinch' in detected_gestures and detected_gestures['pinch']:
                    # Check cooldown period to avoid rapid toggling
                    current_time = time.time()
                    last_time = command_interface.last_command_time.get('pinch', 0)
                    
                    if current_time - last_time > command_interface.command_cooldown:
                        # Toggle annotation mode
                        laser_controller.toggle_annotation()
                        command_interface.last_command_time['pinch'] = current_time
                
                # Handle laser pointer if index finger is pointing
                if 'index_pointing' in detected_gestures and detected_gestures['index_pointing']:
                    # Get index fingertip position (landmark 8)
                    index_tip = gesture_recognizer.get_index_fingertip(landmarks)
                    
                    if index_tip:
                        # Update laser position with smoothing
                        laser_position = laser_controller.update_position(index_tip)
                        laser_active = True
                        
                        # Move the actual system cursor (this will also update the overlay if active)
                        laser_controller.move_cursor(laser_position)
                
                # Process other gestures with command interface
                feedback = command_interface.process_gestures(detected_gestures, landmarks)
                
                # Update presentation state from PowerPoint controller
                presentation_active = ppt_controller.is_presentation_running()
                command_interface.presentation_active = presentation_active
            
            # Add hand detection status
            cv2.putText(hand_frame, "Hand Detected" if hand_present else "No Hand Detected", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add presentation status to the frame
            hand_frame = draw_presentation_status(hand_frame, presentation_active)
            
            # Draw gesture commands info
            hand_frame = draw_gesture_command_info(hand_frame, detected_gestures, command_interface)
            
            # Draw laser pointer on control frame if active
            if laser_position:
                hand_frame = laser_controller.draw_laser(hand_frame, laser_position, laser_active)
            
            # Get current feedback (may be None if expired)
            current_feedback = command_interface.get_feedback()
            
            # Draw command feedback if available
            if current_feedback:
                hand_frame = draw_command_feedback(hand_frame, current_feedback)
            
            # Add annotation mode status
            if laser_controller.annotation_active:
                cv2.putText(hand_frame, "ANNOTATION MODE", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Display the control frame
            cv2.imshow("PowerPoint Presentation Controller", hand_frame)
            
            # Check for exit keys (q or Esc) as backup
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("Key pressed. Stopping...")
                break
            
            # Keyboard shortcuts for testing and control
            if key == ord('n'):  # Next slide
                ppt_controller.next_slide()
            elif key == ord('p'):  # Previous slide
                ppt_controller.previous_slide()
            elif key == ord('s'):  # Start presentation
                ppt_controller.start_presentation()
            elif key == ord('e'):  # End presentation
                ppt_controller.end_presentation()
            elif key == ord('c'):  # Clear annotations
                laser_controller.clear_annotations()
                print("Annotations cleared")
            elif key == ord('a'):  # Toggle annotation mode
                annotation_active = laser_controller.toggle_annotation()
                print(f"Annotation mode {'activated' if annotation_active else 'deactivated'}")
            elif key == ord('l'):  # Toggle laser color
                # Cycle between red, green, and blue
                if laser_controller.laser_color == (255, 0, 0):  # Red
                    laser_controller.laser_color = (0, 255, 0)  # Green
                    laser_controller.overlay.annotation_color = (0, 255, 0)
                    print("Laser color changed to Green")
                elif laser_controller.laser_color == (0, 255, 0):  # Green
                    laser_controller.laser_color = (0, 0, 255)  # Blue
                    laser_controller.overlay.annotation_color = (0, 0, 255)
                    print("Laser color changed to Blue")
                else:  # Blue or any other color
                    laser_controller.laser_color = (255, 0, 0)  # Red
                    laser_controller.overlay.annotation_color = (255, 0, 0)
                    print("Laser color changed to Red")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        exit_flag = True  # Signal terminal thread to exit
        
        if 'processor' in locals():
            processor.release()
        if 'detector' in locals():
            detector.close()
        if 'command_interface' in locals():
            command_interface.shutdown()
        if 'ppt_controller' in locals():
            ppt_controller.shutdown()
        if 'laser_controller' in locals():
            laser_controller.reset()
            
        cv2.destroyAllWindows()
        print("Resources released and windows closed.")
        print("PowerPoint Presentation Controller terminated.")


# Main entry point
def main():
    """Main entry point."""
    global exit_flag
    
    try:
        print("Smart Presentation Controller")
        print("=============================")
        print("Starting PowerPoint controller with laser pointer support...")
        print("The laser pointer will be visible directly on your PowerPoint slides")
        print("Press 'l' to cycle through different laser colors (Red, Green, Blue)")
        print("Press 'a' to toggle annotation mode, 'c' to clear annotations")
        control_powerpoint_with_laser()
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Exiting...")
        

if __name__ == "__main__":
    main()