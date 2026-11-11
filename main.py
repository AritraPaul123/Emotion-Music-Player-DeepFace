import cv2
import numpy as np
from deepface import DeepFace
import customtkinter as ctk
from PIL import Image, ImageTk
from typing import Optional, Dict
import threading
import time
from spotify_auth import SpotifyAuthManager

EMOTION_COLORS = {
    'happy': '#00D26A',      # Vibrant green
    'sad': '#4A90E2',        # Soft blue
    'angry': '#FF4757',      # Bright red
    'surprise': '#FFA502',   # Bright orange
    'neutral': '#A4B0BE',    # Clean cool gray
    'fear': '#9B59B6',       # Purple
    'disgust': '#1ABC9C'     # Teal
}
EMOTION_EMOJIS = {
    'happy': '😊 Happy',
    'sad': '😢 Sad',
    'angry': '😠 Angry',
    'surprise': '😲 Surprised',
    'neutral': '😐 Neutral',
    'fear': '😨 Fearful',
    'disgust': '🤢 Disgusted'
}

class EmotionMusicPlayer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize Spotify
        self.spotify = SpotifyAuthManager()
        
        # Configure window
        self.title("Emotion Music Player")
        self.geometry("800x600")
        
        # Initialize variables
        self.camera = cv2.VideoCapture(0)
        self.current_emotion: Optional[str] = None
        self.last_emotion_change = time.time()
        self.emotion_cooldown = 10  # Seconds between emotion changes
        self.running = True
        
        # Create GUI elements
        self.create_gui()
        
        # Schedule safe UI thread camera updates
        self.after(50, self.update_camera)
    
    def create_gui(self):
        """Create the GUI elements"""
        # Camera frame
        self.camera_frame = ctk.CTkFrame(self, corner_radius=15)
        self.camera_frame.pack(pady=(15, 10), padx=20, fill="both", expand=True)
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera_label.pack(pady=10, padx=10, expand=True)
        
        # Live Emotion Badge directly under the webcam playback
        self.emotion_card = ctk.CTkFrame(self, fg_color="#1e1e24", corner_radius=12, border_width=2, border_color="#3a3a4c")
        self.emotion_card.pack(pady=(0, 15), padx=20, fill="x")
        
        self.emotion_title_label = ctk.CTkLabel(
            self.emotion_card,
            text="LIVE EMOTION",
            font=("Helvetica", 11, "bold"),
            text_color="#9e9eb3"
        )
        self.emotion_title_label.pack(pady=(8, 0))
        
        self.emotion_label = ctk.CTkLabel(
            self.emotion_card, 
            text="Analyzing...",
            font=("Helvetica", 24, "bold"),
            text_color="#A4B0BE"
        )
        self.emotion_label.pack(pady=(2, 4))
        
        self.music_status_label = ctk.CTkLabel(
            self.emotion_card,
            text="🎵 Music syncs on steady expression",
            font=("Helvetica", 12),
            text_color="#7a7a90"
        )
        self.music_status_label.pack(pady=(0, 8))
        
        # Control buttons
        self.control_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.control_frame.pack(pady=(0, 15), padx=20, fill="x")
        
        self.toggle_btn = ctk.CTkButton(
            self.control_frame,
            text="Stop Camera",
            font=("Helvetica", 13, "bold"),
            command=self.toggle_camera
        )
        self.toggle_btn.pack(side="left", padx=10)
        
        self.quit_btn = ctk.CTkButton(
            self.control_frame,
            text="Quit",
            font=("Helvetica", 13, "bold"),
            fg_color="#e03131",
            hover_color="#c92a2a",
            command=self.quit_app
        )
        self.quit_btn.pack(side="right", padx=10)
    
    def update_camera(self):
        """Update camera feed and detect emotions safely via Tkinter loop"""
        if not self.running:
            return

        ret, frame = self.camera.read()
        if ret:
            # Detect emotion
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                emotion = result[0]['dominant_emotion'].lower()
                
                # Update emotion badge dynamically on EVERY frame
                display_text = EMOTION_EMOJIS.get(emotion, emotion.capitalize())
                badge_color = EMOTION_COLORS.get(emotion, "#A4B0BE")
                self.emotion_label.configure(text=display_text, text_color=badge_color)
                self.emotion_card.configure(border_color=badge_color)
                
                # Music playback triggers only after emotion is stable and cooldown passed
                current_time = time.time()
                if (emotion != self.current_emotion and 
                    current_time - self.last_emotion_change > self.emotion_cooldown):
                    self.current_emotion = emotion
                    self.last_emotion_change = current_time
                    self.music_status_label.configure(text=f"🎶 Now playing music for: {emotion.capitalize()}")
                    print(f"Detected Emotion: {emotion}. Triggering Spotify playback...")
                    threading.Thread(
                        target=self.spotify.play_tracks_by_mood,
                        args=(emotion,),
                        daemon=True
                    ).start()
            
            except Exception as e:
                print(f"Error detecting emotion: {str(e)}")
            
            # Convert frame for display
            disp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            disp_frame = cv2.resize(disp_frame, (640, 480))
            img = Image.fromarray(disp_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.camera_label.configure(image=img_tk)
            self.camera_label.image = img_tk
        
        if self.running:
            self.after(30, self.update_camera)
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.toggle_btn.cget("text") == "Stop":
            self.toggle_btn.configure(text="Start")
            self.running = False
        else:
            self.toggle_btn.configure(text="Stop")
            self.running = True
            self.after(30, self.update_camera)
    
    def quit_app(self):
        """Clean up and quit application"""
        self.running = False
        if self.camera.isOpened():
            self.camera.release()
        self.quit()

if __name__ == "__main__":
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run app
    app = EmotionMusicPlayer()
    app.mainloop()