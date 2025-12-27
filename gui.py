import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import sys
import os

# Add current dir to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run_pipeline

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Video Dubbing Pro")
        self.geometry("700x600")

        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- File Input ---
        self.lbl_input = ctk.CTkLabel(self, text="Input Video:")
        self.lbl_input.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        self.entry_input = ctk.CTkEntry(self, placeholder_text="Select video file...")
        self.entry_input.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="ew")
        
        self.btn_browse_input = ctk.CTkButton(self, text="Browse", width=100, command=self.browse_input)
        self.btn_browse_input.grid(row=0, column=2, padx=20, pady=(20, 10))

        # --- File Output ---
        self.lbl_output = ctk.CTkLabel(self, text="Output Video:")
        self.lbl_output.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        
        self.entry_output = ctk.CTkEntry(self, placeholder_text="Save as...")
        self.entry_output.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        
        self.btn_browse_output = ctk.CTkButton(self, text="Browse", width=100, command=self.browse_output)
        self.btn_browse_output.grid(row=1, column=2, padx=20, pady=10)

        # --- Settings Frame ---
        self.frame_settings = ctk.CTkFrame(self)
        self.frame_settings.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        self.frame_settings.grid_columnconfigure(1, weight=1)
        
        # Male Pitch
        self.lbl_male_pitch = ctk.CTkLabel(self.frame_settings, text="Male Tone Adjuctment (Pitch):")
        self.lbl_male_pitch.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.slider_male = ctk.CTkSlider(self.frame_settings, from_=-10, to=10, number_of_steps=20)
        self.slider_male.set(0)
        self.slider_male.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.lbl_male_val = ctk.CTkLabel(self.frame_settings, text="0.0")
        self.lbl_male_val.grid(row=0, column=2, padx=10, pady=10)
        self.slider_male.configure(command=lambda val: self.lbl_male_val.configure(text=f"{val:.1f}"))

        # Female Pitch
        self.lbl_fem_pitch = ctk.CTkLabel(self.frame_settings, text="Female Tone Adjustment (Pitch):")
        self.lbl_fem_pitch.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.slider_fem = ctk.CTkSlider(self.frame_settings, from_=-10, to=10, number_of_steps=20)
        self.slider_fem.set(0)
        self.slider_fem.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.lbl_fem_val = ctk.CTkLabel(self.frame_settings, text="0.0")
        self.lbl_fem_val.grid(row=1, column=2, padx=10, pady=10)
        self.slider_fem.configure(command=lambda val: self.lbl_fem_val.configure(text=f"{val:.1f}"))

        # --- Logs ---
        self.textbox_log = ctk.CTkTextbox(self, width=600, height=200)
        self.textbox_log.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky="nsew")
        
        # --- Progress & Action ---
        self.progressbar = ctk.CTkProgressBar(self)
        self.progressbar.grid(row=4, column=0, columnspan=3, padx=20, pady=(10, 0), sticky="ew")
        self.progressbar.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Ready")
        self.lbl_status.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="w")

        self.btn_start = ctk.CTkButton(self, text="START DUBBING", font=("Arial", 16, "bold"), height=40, command=self.start_thread)
        self.btn_start.grid(row=5, column=2, padx=(10, 20), pady=10, sticky="ew")

        self.btn_stop = ctk.CTkButton(self, text="STOP", font=("Arial", 16, "bold"), height=40, fg_color="#D32F2F", hover_color="#B71C1C", command=self.stop_process, state="disabled")
        self.btn_stop.grid(row=5, column=1, padx=(20, 10), pady=10, sticky="ew")

        self.stop_event = None

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov")])
        if filename:
            self.entry_input.delete(0, "end")
            self.entry_input.insert(0, filename)
            # Auto set output
            if not self.entry_output.get():
                base, ext = os.path.splitext(filename)
                self.entry_output.insert(0, f"{base}_dubbed_th{ext}")

    def browse_output(self):
        filename = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 file", "*.mp4")])
        if filename:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, filename)

    def log(self, message):
        self.textbox_log.insert("end", message + "\n")
        self.textbox_log.see("end")

    def context_log_wrapper(self, msg):
        # Allow thread to update UI safely
        self.after(0, lambda: self.log(msg))

    def context_progress_wrapper(self, val, msg):
        self.after(0, lambda: self.update_progress(val, msg))

    def update_progress(self, val, msg):
        self.progressbar.set(val)
        self.lbl_status.configure(text=msg)

    def start_thread(self):
        input_file = self.entry_input.get()
        output_file = self.entry_output.get()
        
        if not input_file:
            self.log("Error: Please select an input file.")
            return

        if not output_file:
            self.log("Error: Please select an output file.")
            return

        male_pitch = self.slider_male.get()
        female_pitch = self.slider_fem.get()
        
        self.btn_start.configure(state="disabled", text="Running...")
        self.btn_stop.configure(state="normal")
        self.progressbar.set(0)
        self.textbox_log.delete("1.0", "end")
        
        self.stop_event = threading.Event()
        
        thread = threading.Thread(target=self.run_process, args=(input_file, output_file, male_pitch, female_pitch))
        thread.start()

    def stop_process(self):
        if self.stop_event:
            self.stop_event.set()
            self.log("Stopping... please wait for current step to finish.")
            self.btn_stop.configure(state="disabled")

    def run_process(self, input_file, output_file, male_pitch, female_pitch):
        try:
            success = run_pipeline(
                input_file, 
                output_file, 
                language="th",
                progress_callback=self.context_progress_wrapper,
                log_callback=self.context_log_wrapper,
                male_pitch=male_pitch,
                female_pitch=female_pitch,
                stop_event=self.stop_event
            )
            if success:
                self.after(0, lambda: self.log("SUCCESS: Dubbing complete!"))
            else:
                 self.after(0, lambda: self.log("Process Stopped/Failed."))
        except Exception as e:
            self.after(0, lambda: self.log(f"FAILURE: {e}"))
        finally:
            self.after(0, lambda: self.reset_buttons())
            
    def reset_buttons(self):
        self.btn_start.configure(state="normal", text="START DUBBING")
        self.btn_stop.configure(state="disabled")

if __name__ == "__main__":
    app = App()
    app.mainloop()
