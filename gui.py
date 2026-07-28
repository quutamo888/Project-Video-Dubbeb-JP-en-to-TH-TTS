import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import sys
import os
import psutil

# Monkey-patch torchaudio.load BEFORE importing modules that use it
# This avoids torchcodec issues on Windows with PyTorch nightly
try:
    import torch
    import torchaudio
    import soundfile as sf
    import librosa
    import numpy as np
    
    _original_torchaudio_load = torchaudio.load
    
    def _patched_torchaudio_load(filepath, *args, **kwargs):
        try:
            waveform, sample_rate = sf.read(filepath, dtype='float32')
            waveform = torch.from_numpy(waveform)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.T
            return waveform, sample_rate
        except Exception:
            try:
                waveform, sample_rate = librosa.load(filepath, sr=None, mono=False)
                waveform = torch.from_numpy(waveform)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                return waveform, sample_rate
            except Exception:
                return _original_torchaudio_load(filepath, *args, **kwargs)
    
    torchaudio.load = _patched_torchaudio_load
except Exception:
    pass

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Add current dir to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run_pipeline

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Video Dubbing Pro")
        self.title("AI Video Dubbing Pro")
        self.geometry("700x750")

        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)

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

        # Translation Temperature
        self.lbl_temp = ctk.CTkLabel(self.frame_settings, text="Translation Creativity (Temperature):")
        self.lbl_temp.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.slider_temp = ctk.CTkSlider(self.frame_settings, from_=0.1, to=1.0, number_of_steps=9)
        self.slider_temp.set(0.3)
        self.slider_temp.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.lbl_temp_val = ctk.CTkLabel(self.frame_settings, text="0.3")
        self.lbl_temp_val.grid(row=2, column=2, padx=10, pady=10)
        self.slider_temp.configure(command=lambda val: self.lbl_temp_val.configure(text=f"{val:.1f}"))

        # Translation Provider
        self.lbl_translator = ctk.CTkLabel(self.frame_settings, text="Translation Provider:")
        self.lbl_translator.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.combo_translator = ctk.CTkComboBox(
            self.frame_settings, 
            values=["ollama (Flexible, needs Ollama)", "local-transformer (Fast, Local)", "google (Fast, Online)"],
            state="readonly"
        )
        self.combo_translator.set("ollama (Flexible, needs Ollama)")
        self.combo_translator.grid(row=3, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # TTS Provider
        self.lbl_tts = ctk.CTkLabel(self.frame_settings, text="Voice Generator (TTS):")
        self.lbl_tts.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.combo_tts = ctk.CTkComboBox(
            self.frame_settings,
            values=["omnivoice (Voice Clone, Natural)", "f5-tts (Voice Clone, Slow)", "edge-tts (Natural, Fast)", "mms (Basic, Local)"],
            state="readonly"
        )
        self.combo_tts.set("omnivoice (Voice Clone, Natural)")
        self.combo_tts.grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Gender Detection Method
        self.lbl_gender = ctk.CTkLabel(self.frame_settings, text="Gender Detection:")
        self.lbl_gender.grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.combo_gender = ctk.CTkComboBox(
            self.frame_settings,
            values=["audio (Fast, Pitch)", "visual (Accurate, Face)", "hybrid (Best, Slow)"],
            state="readonly"
        )
        self.combo_gender.set("audio (Fast, Pitch)")
        self.combo_gender.grid(row=5, column=1, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Audio Gender Model (ML vs Pitch)
        self.lbl_audio_model = ctk.CTkLabel(self.frame_settings, text="Audio Model:")
        self.lbl_audio_model.grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.combo_audio_model = ctk.CTkComboBox(
            self.frame_settings,
            values=["ml (AI, Accurate)", "pitch (Hz, Fast)"],
            state="readonly"
        )
        self.combo_audio_model.set("ml (AI, Accurate)")
        self.combo_audio_model.grid(row=6, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

        # --- System Monitor ---
        self.frame_monitor = ctk.CTkFrame(self)
        self.frame_monitor.grid(row=7, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        self.frame_monitor.grid_columnconfigure(1, weight=1)
        
        # Header
        self.lbl_mon_title = ctk.CTkLabel(self.frame_monitor, text="System Resources", font=("Arial", 12, "bold"))
        self.lbl_mon_title.grid(row=0, column=0, columnspan=3, pady=(5,0))

        # CPU
        self.lbl_cpu = ctk.CTkLabel(self.frame_monitor, text="CPU Usage:")
        self.lbl_cpu.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.prog_cpu = ctk.CTkProgressBar(self.frame_monitor)
        self.prog_cpu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.prog_cpu.set(0)
        self.lbl_cpu_val = ctk.CTkLabel(self.frame_monitor, text="0%")
        self.lbl_cpu_val.grid(row=1, column=2, padx=10, pady=5)

        # GPU
        self.lbl_gpu = ctk.CTkLabel(self.frame_monitor, text="GPU Usage:")
        self.lbl_gpu.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.prog_gpu = ctk.CTkProgressBar(self.frame_monitor)
        self.prog_gpu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.prog_gpu.set(0)
        self.lbl_gpu_val = ctk.CTkLabel(self.frame_monitor, text="0%")
        self.lbl_gpu_val.grid(row=2, column=2, padx=10, pady=5)
        
        # Disk (Project Drive)
        self.lbl_disk = ctk.CTkLabel(self.frame_monitor, text="Disk Space:")
        self.lbl_disk.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.prog_disk = ctk.CTkProgressBar(self.frame_monitor)
        self.prog_disk.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.prog_disk.set(0)
        self.lbl_disk_val = ctk.CTkLabel(self.frame_monitor, text="0%")
        self.lbl_disk_val.grid(row=3, column=2, padx=10, pady=5)
        
        # Start Monitor Thread
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

        # --- Logs ---
        self.textbox_log = ctk.CTkTextbox(self, width=600, height=200)
        self.textbox_log.grid(row=4, column=0, columnspan=3, padx=20, pady=10, sticky="nsew")
        
        # --- Progress & Action ---
        self.progressbar = ctk.CTkProgressBar(self)
        self.progressbar.grid(row=5, column=0, columnspan=3, padx=20, pady=(10, 0), sticky="ew")
        self.progressbar.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Ready")
        self.lbl_status.grid(row=6, column=0, columnspan=2, padx=20, pady=10, sticky="w")

        self.btn_start = ctk.CTkButton(self, text="START DUBBING", font=("Arial", 16, "bold"), height=40, command=self.start_thread)
        self.btn_start.grid(row=6, column=2, padx=(10, 20), pady=10, sticky="ew")

        self.btn_stop = ctk.CTkButton(self, text="STOP", font=("Arial", 16, "bold"), height=40, fg_color="#D32F2F", hover_color="#B71C1C", command=self.stop_process, state="disabled")
        self.btn_stop.grid(row=6, column=1, padx=(20, 10), pady=10, sticky="ew")

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
        temperature = self.slider_temp.get()
        
        # Extract provider selections
        translator_choice = self.combo_translator.get()
        tts_choice = self.combo_tts.get()
        gender_choice = self.combo_gender.get()
        audio_model_choice = self.combo_audio_model.get()
        
        # Map friendly names to provider codes
        translator_map = {
            "ollama (Flexible, needs Ollama)": "ollama",
            "local-transformer (Fast, Local)": "local-transformer",
            "google (Fast, Online)": "google"
        }
        tts_map = {
            "omnivoice (Voice Clone, Natural)": "omnivoice",
            "f5-tts (Voice Clone, Slow)": "f5-tts",
            "edge-tts (Natural, Fast)": "edge-tts",
            "mms (Basic, Local)": "mms"
        }
        gender_map = {
            "audio (Fast, Pitch)": "audio",
            "visual (Accurate, Face)": "visual",
            "hybrid (Best, Slow)": "hybrid"
        }
        audio_model_map = {
            "ml (AI, Accurate)": "ml",
            "pitch (Hz, Fast)": "pitch"
        }
        
        translator_provider = translator_map.get(translator_choice, "ollama")
        tts_provider = tts_map.get(tts_choice, "edge-tts")
        gender_method = gender_map.get(gender_choice, "audio")
        audio_model = audio_model_map.get(audio_model_choice, "ml")
        
        self.btn_start.configure(state="disabled", text="Running...")
        self.btn_stop.configure(state="normal")
        self.progressbar.set(0)
        self.textbox_log.delete("1.0", "end")
        
        self.stop_event = threading.Event()
        
        thread = threading.Thread(
            target=self.run_process, 
            args=(input_file, output_file, male_pitch, female_pitch, temperature, translator_provider, tts_provider, gender_method, audio_model)
        )
        thread.start()

    def stop_process(self):
        if self.stop_event:
            self.stop_event.set()
            self.log("Stopping... please wait for current step to finish.")
            self.btn_stop.configure(state="disabled")

    def run_process(self, input_file, output_file, male_pitch, female_pitch, temperature, translator_provider, tts_provider, gender_method, audio_model):
        # Temporarily override Config settings with GUI selections
        from config import Config
        original_translator = Config.TRANSLATION_PROVIDER
        original_tts = Config.TTS_PROVIDER
        original_gender = Config.GENDER_DETECTION_METHOD
        original_audio_model = Config.AUDIO_GENDER_MODEL
        
        try:
            # Apply GUI selections
            Config.TRANSLATION_PROVIDER = translator_provider
            Config.TTS_PROVIDER = tts_provider
            Config.GENDER_DETECTION_METHOD = gender_method
            Config.AUDIO_GENDER_MODEL = audio_model
            
            success = run_pipeline(
                input_file, 
                output_file, 
                language="th",
                progress_callback=self.context_progress_wrapper,
                log_callback=self.context_log_wrapper,
                male_pitch=male_pitch,
                female_pitch=female_pitch,
                translation_temperature=temperature,
                stop_event=self.stop_event
            )
            if success:
                self.after(0, lambda: self.log("SUCCESS: Dubbing complete!"))
            else:
                 self.after(0, lambda: self.log("Process Stopped/Failed."))
        except Exception as e:
            err_msg = f"FAILURE: {e}"
            self.after(0, lambda: self.log(err_msg))
        finally:
            # Restore original config
            Config.TRANSLATION_PROVIDER = original_translator
            Config.TTS_PROVIDER = original_tts
            Config.GENDER_DETECTION_METHOD = original_gender
            Config.AUDIO_GENDER_MODEL = original_audio_model
            self.after(0, lambda: self.reset_buttons())
            
    def reset_buttons(self):
        self.btn_start.configure(state="normal", text="START DUBBING")
        self.btn_stop.configure(state="disabled")

    def monitor_loop(self):
        import time
        while self.monitor_active:
            try:
                # CPU
                cpu = psutil.cpu_percent(interval=0.5)
                
                # GPU
                gpu = 0
                if HAS_GPUTIL:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0].load * 100
                
                # Disk (Usage of CWD drive)
                cwd = os.getcwd()
                disk = psutil.disk_usage(cwd).percent
                
                # Update UI
                self.after(0, lambda c=cpu, g=gpu, d=disk: self.update_monitor(c, g, d))
                
                time.sleep(0.5)
            except Exception as e:
                # Silent fail to not crash UI
                pass
                
    def update_monitor(self, cpu, gpu, disk):
        self.prog_cpu.set(cpu / 100)
        self.lbl_cpu_val.configure(text=f"{cpu:.1f}%")
        
        self.prog_gpu.set(gpu / 100)
        self.lbl_gpu_val.configure(text=f"{gpu:.1f}%")
        
        self.prog_disk.set(disk / 100)
        self.lbl_disk_val.configure(text=f"{disk:.1f}%")

    def on_closing(self):
        self.monitor_active = False
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
