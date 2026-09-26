import sys
import os

print("Starting AI Video Dubbing Pro...", flush=True)

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import psutil

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

# Add current dir to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import setup_logging
import logging
setup_logging(Config.LOGS_DIR)
gui_logger = logging.getLogger("gui")

def _apply_torchaudio_patch():
    try:
        import torch
        import torchaudio
        import soundfile as sf
        import librosa
        
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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SubtitleTranslatorWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("📄 Standalone Subtitle Translator (Offline GPU)")
        self.geometry("650x580")
        self.resizable(False, False)
        
        self.grid_columnconfigure(1, weight=1)
        
        # Title Header
        self.lbl_title = ctk.CTkLabel(
            self, 
            text="Offline Subtitle Translator (.srt / .vtt)", 
            font=("Arial", 16, "bold")
        )
        self.lbl_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(15, 10))
        
        # Subtitle Input File
        self.lbl_sub_in = ctk.CTkLabel(self, text="Input Subtitle:")
        self.lbl_sub_in.grid(row=1, column=0, padx=20, pady=8, sticky="w")
        self.entry_sub_in = ctk.CTkEntry(self, placeholder_text="Select Japanese or other .srt file...")
        self.entry_sub_in.grid(row=1, column=1, padx=(0, 10), pady=8, sticky="ew")
        self.btn_sub_in = ctk.CTkButton(self, text="Browse", width=80, command=self.browse_sub_in)
        self.btn_sub_in.grid(row=1, column=2, padx=(0, 20), pady=8)
        
        # Subtitle Output File
        self.lbl_sub_out = ctk.CTkLabel(self, text="Output Subtitle:")
        self.lbl_sub_out.grid(row=2, column=0, padx=20, pady=8, sticky="w")
        self.entry_sub_out = ctk.CTkEntry(self, placeholder_text="Save translated .srt file as...")
        self.entry_sub_out.grid(row=2, column=1, padx=(0, 10), pady=8, sticky="ew")
        self.btn_sub_out = ctk.CTkButton(self, text="Browse", width=80, command=self.browse_sub_out)
        self.btn_sub_out.grid(row=2, column=2, padx=(0, 20), pady=8)
        
        # Source & Target Languages Frame
        self.frame_langs = ctk.CTkFrame(self)
        self.frame_langs.grid(row=3, column=0, columnspan=3, padx=20, pady=8, sticky="ew")
        self.frame_langs.grid_columnconfigure(1, weight=1)
        self.frame_langs.grid_columnconfigure(3, weight=1)
        
        self.lbl_lang_src = ctk.CTkLabel(self.frame_langs, text="Source:")
        self.lbl_lang_src.grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.combo_lang_src = ctk.CTkComboBox(
            self.frame_langs,
            values=["Japanese (ja)", "English (en)", "Chinese (zh)", "Korean (ko)"],
            state="readonly"
        )
        self.combo_lang_src.set("Japanese (ja)")
        self.combo_lang_src.grid(row=0, column=1, padx=10, pady=8, sticky="ew")
        
        self.lbl_lang_tgt = ctk.CTkLabel(self.frame_langs, text="Target:")
        self.lbl_lang_tgt.grid(row=0, column=2, padx=10, pady=8, sticky="w")
        self.combo_lang_tgt = ctk.CTkComboBox(
            self.frame_langs,
            values=["Thai (th)", "English (en)"],
            state="readonly"
        )
        self.combo_lang_tgt.set("Thai (th)")
        self.combo_lang_tgt.grid(row=0, column=3, padx=10, pady=8, sticky="ew")

        # Engine Selection Frame
        self.frame_engine = ctk.CTkFrame(self)
        self.frame_engine.grid(row=4, column=0, columnspan=3, padx=20, pady=(2, 6), sticky="ew")
        self.frame_engine.grid_columnconfigure(1, weight=1)

        self.lbl_engine = ctk.CTkLabel(self.frame_engine, text="Engine:", font=("Arial", 12, "bold"))
        self.lbl_engine.grid(row=0, column=0, padx=10, pady=6, sticky="w")

        self.combo_sub_engine = ctk.CTkComboBox(
            self.frame_engine,
            values=[
                "Qwen2.5-3B (Context-Aware LLM, Best Thai Quality, Offline GPU)",
                "CTranslate2 NLLB-200 (Fastest GPU, Offline)"
            ],
            state="readonly"
        )
        self.combo_sub_engine.set("Qwen2.5-3B (Context-Aware LLM, Best Thai Quality, Offline GPU)")
        self.combo_sub_engine.grid(row=0, column=1, padx=10, pady=6, sticky="ew")
        
        # Logs
        self.textbox_sub_log = ctk.CTkTextbox(self, width=600, height=180)
        self.textbox_sub_log.grid(row=5, column=0, columnspan=3, padx=20, pady=5, sticky="nsew")
        
        # Progress Bar & Status
        self.prog_sub = ctk.CTkProgressBar(self)
        self.prog_sub.grid(row=6, column=0, columnspan=3, padx=20, pady=(10, 5), sticky="ew")
        self.prog_sub.set(0)
        
        self.lbl_sub_status = ctk.CTkLabel(self, text="Ready")
        self.lbl_sub_status.grid(row=7, column=0, columnspan=2, padx=20, pady=10, sticky="w")
        
        # Action Buttons
        self.btn_sub_start = ctk.CTkButton(
            self, 
            text="TRANSLATE SUBTITLES", 
            font=("Arial", 13, "bold"),
            fg_color="#2E7D32", 
            hover_color="#1B5E20",
            height=36,
            command=self.start_translation
        )
        self.btn_sub_start.grid(row=7, column=2, padx=(0, 20), pady=10, sticky="e")

    def browse_sub_in(self):
        filename = filedialog.askopenfilename(filetypes=[("Subtitle files", "*.srt *.vtt")])
        if filename:
            self.entry_sub_in.delete(0, "end")
            self.entry_sub_in.insert(0, filename)
            if not self.entry_sub_out.get():
                base, ext = os.path.splitext(filename)
                self.entry_sub_out.insert(0, f"{base}.th{ext}")

    def browse_sub_out(self):
        filename = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=[("Subtitle files", "*.srt *.vtt")])
        if filename:
            self.entry_sub_out.delete(0, "end")
            self.entry_sub_out.insert(0, filename)

    def log(self, msg):
        self.textbox_sub_log.insert("end", f"{msg}\n")
        self.textbox_sub_log.see("end")

    def start_translation(self):
        in_path = self.entry_sub_in.get().strip()
        out_path = self.entry_sub_out.get().strip()
        
        if not in_path or not os.path.exists(in_path):
            self.lbl_sub_status.configure(text="Error: Input subtitle file not found.")
            return
            
        src_map = {"Japanese (ja)": "ja", "English (en)": "en", "Chinese (zh)": "zh", "Korean (ko)": "ko"}
        tgt_map = {"Thai (th)": "th", "English (en)": "en"}
        
        src_code = src_map.get(self.combo_lang_src.get(), "ja")
        tgt_code = tgt_map.get(self.combo_lang_tgt.get(), "th")
        engine_choice = self.combo_sub_engine.get()
        engine_code = "qwen" if "qwen" in engine_choice.lower() else "ctranslate2"

        self.btn_sub_start.configure(state="disabled", text="Translating...")
        self.prog_sub.set(0)
        self.textbox_sub_log.delete("1.0", "end")

        def run():
            from modules.translator import translate_srt_file
            try:
                def prog_cb(pct, status_text):
                    self.after(0, lambda: self.prog_sub.set(pct))
                    self.after(0, lambda: self.lbl_sub_status.configure(text=status_text))

                def log_cb(msg):
                    self.after(0, lambda: self.log(msg))

                res = translate_srt_file(in_path, out_path, source_lang=src_code, target_lang=tgt_code, engine=engine_code, progress_callback=prog_cb, log_callback=log_cb)
                self.after(0, lambda: self.lbl_sub_status.configure(text=f"✅ Done: {os.path.basename(res)}"))
            except Exception as e:
                self.after(0, lambda: self.log(f"❌ Translation failed: {e}"))
                self.after(0, lambda: self.lbl_sub_status.configure(text="Failed"))
            finally:
                self.after(0, lambda: self.btn_sub_start.configure(state="normal", text="TRANSLATE SUBTITLES"))

        threading.Thread(target=run, daemon=True).start()

class YouTubeDownloaderWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("⬇️ YouTube Video Downloader")
        self.geometry("620x520")
        self.resizable(False, False)

        self.grid_columnconfigure(1, weight=1)

        # Title Header
        self.lbl_title = ctk.CTkLabel(
            self,
            text="🎬 YouTube Video Downloader",
            font=("Arial", 16, "bold")
        )
        self.lbl_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(15, 8))

        # URL Input
        self.lbl_url = ctk.CTkLabel(self, text="YouTube URL:")
        self.lbl_url.grid(row=1, column=0, padx=20, pady=6, sticky="w")
        self.entry_url = ctk.CTkEntry(self, placeholder_text="https://www.youtube.com/watch?v=...")
        self.entry_url.grid(row=1, column=1, columnspan=2, padx=(0, 20), pady=6, sticky="ew")

        # Resolution Selection
        self.lbl_res = ctk.CTkLabel(self, text="Quality (Resolution):")
        self.lbl_res.grid(row=2, column=0, padx=20, pady=6, sticky="w")
        self.combo_res = ctk.CTkComboBox(
            self,
            values=[
                "1080p (Full HD, คมชัดสูง)",
                "720p (HD, แนะนำ)",
                "480p (SD, ขนาดปานกลาง)",
                "360p (Fast, ประหยัดขนาด)",
                "Best Available (สูงสุดเท่าที่มี)"
            ],
            state="readonly"
        )
        self.combo_res.set("1080p (Full HD, คมชัดสูง)")
        self.combo_res.grid(row=2, column=1, columnspan=2, padx=(0, 20), pady=6, sticky="ew")

        # Save Directory
        self.lbl_save_dir = ctk.CTkLabel(self, text="Save Folder:")
        self.lbl_save_dir.grid(row=3, column=0, padx=20, pady=6, sticky="w")
        default_dir = os.path.abspath("input")
        os.makedirs(default_dir, exist_ok=True)
        self.entry_save_dir = ctk.CTkEntry(self)
        self.entry_save_dir.insert(0, default_dir)
        self.entry_save_dir.grid(row=3, column=1, padx=(0, 8), pady=6, sticky="ew")
        self.btn_browse_dir = ctk.CTkButton(self, text="Browse", width=75, command=self.browse_dir)
        self.btn_browse_dir.grid(row=3, column=2, padx=(0, 20), pady=6)

        # Auto-load checkbox
        self.chk_auto_load = ctk.CTkCheckBox(self, text="Auto-set as Input Video for Dubbing Pipeline")
        self.chk_auto_load.select()
        self.chk_auto_load.grid(row=4, column=0, columnspan=3, padx=20, pady=6, sticky="w")

        # Progress Bar & Status
        self.prog_dl = ctk.CTkProgressBar(self)
        self.prog_dl.grid(row=5, column=0, columnspan=3, padx=20, pady=(8, 2), sticky="ew")
        self.prog_dl.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.lbl_status.grid(row=6, column=0, columnspan=3, padx=20, pady=(0, 4), sticky="w")

        # Action Buttons
        self.frame_btns = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_btns.grid(row=7, column=0, columnspan=3, padx=20, pady=4, sticky="ew")
        self.frame_btns.grid_columnconfigure(0, weight=1)
        self.frame_btns.grid_columnconfigure(1, weight=2)

        self.btn_close = ctk.CTkButton(
            self.frame_btns,
            text="Close",
            fg_color="#546E7A",
            hover_color="#37474F",
            command=self.destroy
        )
        self.btn_close.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.btn_start = ctk.CTkButton(
            self.frame_btns,
            text="DOWNLOAD VIDEO",
            font=("Arial", 13, "bold"),
            fg_color="#C62828",
            hover_color="#8E0000",
            command=self.start_download
        )
        self.btn_start.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        # Log textbox
        self.textbox_log = ctk.CTkTextbox(self, height=130)
        self.textbox_log.grid(row=8, column=0, columnspan=3, padx=20, pady=(4, 15), sticky="nsew")
        self.grid_rowconfigure(8, weight=1)

    def browse_dir(self):
        folder = filedialog.askdirectory(initialdir=self.entry_save_dir.get())
        if folder:
            self.entry_save_dir.delete(0, "end")
            self.entry_save_dir.insert(0, folder)

    def log(self, msg):
        self.textbox_log.insert("end", f"{msg}\n")
        self.textbox_log.see("end")

    def start_download(self):
        url = self.entry_url.get().strip()
        if not url:
            self.lbl_status.configure(text="Error: Please enter a YouTube URL.")
            return

        res_choice = self.combo_res.get()
        if "1080" in res_choice:
            res_code = "1080p"
        elif "720" in res_choice:
            res_code = "720p"
        elif "480" in res_choice:
            res_code = "480p"
        elif "360" in res_choice:
            res_code = "360p"
        else:
            res_code = "best"

        save_dir = self.entry_save_dir.get().strip() or "input"
        auto_load = self.chk_auto_load.get() == 1

        self.btn_start.configure(state="disabled", text="Downloading...")
        self.prog_dl.set(0)
        self.textbox_log.delete("1.0", "end")

        def run():
            from modules.youtube_downloader import download_youtube_video
            try:
                def prog_cb(pct, status_text):
                    self.after(0, lambda: self.prog_dl.set(pct))
                    self.after(0, lambda: self.lbl_status.configure(text=status_text))

                def log_cb(msg):
                    self.after(0, lambda: self.log(msg))

                downloaded_file = download_youtube_video(
                    url=url,
                    resolution=res_code,
                    output_dir=save_dir,
                    progress_callback=prog_cb,
                    log_callback=log_cb
                )

                self.after(0, lambda: self.lbl_status.configure(text=f"✅ Download complete: {os.path.basename(downloaded_file)}"))
                if auto_load and hasattr(self.parent, 'entry_input'):
                    def update_parent():
                        self.parent.entry_input.delete(0, "end")
                        self.parent.entry_input.insert(0, downloaded_file)
                        base, ext = os.path.splitext(downloaded_file)
                        self.parent.entry_output.delete(0, "end")
                        self.parent.entry_output.insert(0, f"{base}_dubbed_th{ext}")
                        self.parent.log(f"📥 Downloaded YouTube video loaded into Input: {downloaded_file}")
                    self.after(0, update_parent)

            except Exception as e:
                self.after(0, lambda: self.log(f"❌ Download failed: {e}"))
                self.after(0, lambda: self.lbl_status.configure(text="Download Failed"))
            finally:
                self.after(0, lambda: self.btn_start.configure(state="normal", text="DOWNLOAD VIDEO"))

        threading.Thread(target=run, daemon=True).start()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Video Dubbing Pro")
        self.geometry("1300x760")
        self.minsize(1120, 660)

        # 3-Column Grid Layout
        self.grid_columnconfigure(0, weight=3)  # Col 1: Sources & Translation
        self.grid_columnconfigure(1, weight=3)  # Col 2: Voice & AI Engine
        self.grid_columnconfigure(2, weight=4)  # Col 3: Monitor, Logs & Actions
        self.grid_rowconfigure(0, weight=1)

        # =========================================================
        # COLUMN 1: Sources, Files & Translation
        # =========================================================
        self.col1 = ctk.CTkScrollableFrame(self, label_text="📁 Sources & Translation")
        self.col1.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
        self.col1.grid_columnconfigure(1, weight=1)

        # --- 1. File Input / Output ---
        self.frame_files = ctk.CTkFrame(self.col1)
        self.frame_files.pack(fill="x", padx=4, pady=(4, 8))
        self.frame_files.grid_columnconfigure(1, weight=1)

        self.lbl_files_hdr = ctk.CTkLabel(self.frame_files, text="📁 File Selection", font=("Arial", 12, "bold"))
        self.lbl_files_hdr.grid(row=0, column=0, columnspan=4, padx=10, pady=(8, 4), sticky="w")

        # Input Video
        self.lbl_input = ctk.CTkLabel(self.frame_files, text="Input Video:")
        self.lbl_input.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_input = ctk.CTkEntry(self.frame_files, placeholder_text="Select video file or fetch via YouTube...")
        self.entry_input.grid(row=1, column=1, padx=(0, 6), pady=5, sticky="ew")
        self.btn_browse_input = ctk.CTkButton(self.frame_files, text="Browse", width=65, command=self.browse_input)
        self.btn_browse_input.grid(row=1, column=2, padx=(0, 4), pady=5)
        self.btn_yt_dl = ctk.CTkButton(
            self.frame_files,
            text="⬇️ YouTube",
            width=78,
            fg_color="#C62828",
            hover_color="#8E0000",
            command=self.open_youtube_downloader
        )
        self.btn_yt_dl.grid(row=1, column=3, padx=(0, 8), pady=5)

        # Output Video
        self.lbl_output = ctk.CTkLabel(self.frame_files, text="Output Video:")
        self.lbl_output.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.entry_output = ctk.CTkEntry(self.frame_files, placeholder_text="Save as...")
        self.entry_output.grid(row=2, column=1, padx=(0, 6), pady=5, sticky="ew")
        self.btn_browse_output = ctk.CTkButton(self.frame_files, text="Browse", width=65, command=self.browse_output)
        self.btn_browse_output.grid(row=2, column=2, columnspan=2, padx=(0, 8), pady=5, sticky="ew")

        # --- 2. Translation & Language ---
        self.frame_trans = ctk.CTkFrame(self.col1)
        self.frame_trans.pack(fill="x", padx=4, pady=4)
        self.frame_trans.grid_columnconfigure(1, weight=1)

        self.lbl_trans_hdr = ctk.CTkLabel(self.frame_trans, text="🌐 Speech-to-Text & Translation", font=("Arial", 12, "bold"))
        self.lbl_trans_hdr.grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w")

        # STT Engine
        self.lbl_stt = ctk.CTkLabel(self.frame_trans, text="STT Engine:")
        self.lbl_stt.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combo_stt = ctk.CTkComboBox(
            self.frame_trans,
            values=["Kotoba-Whisper (Japanese SOTA, Ultra Accurate)", "faster-whisper (Large-v2, Universal)"],
            state="readonly"
        )
        self.combo_stt.set("Kotoba-Whisper (Japanese SOTA, Ultra Accurate)")
        self.combo_stt.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Source Audio Lang
        self.lbl_source_lang = ctk.CTkLabel(self.frame_trans, text="Source Audio:")
        self.lbl_source_lang.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.combo_source_lang = ctk.CTkComboBox(
            self.frame_trans,
            values=["Auto Detect (auto)", "Japanese (ja)", "Chinese (zh)", "Korean (ko)", "English (en)", "Thai (th)"],
            state="readonly"
        )
        self.combo_source_lang.set("Auto Detect (auto)")
        self.combo_source_lang.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Target Subtitle Lang
        self.lbl_target_lang = ctk.CTkLabel(self.frame_trans, text="Subtitle Output:")
        self.lbl_target_lang.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.combo_target_lang = ctk.CTkComboBox(
            self.frame_trans,
            values=["Thai (th)", "English (en)"],
            state="readonly"
        )
        self.combo_target_lang.set("Thai (th)")
        self.combo_target_lang.grid(row=3, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Translation Provider
        self.lbl_translator = ctk.CTkLabel(self.frame_trans, text="Provider:")
        self.lbl_translator.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.combo_translator = ctk.CTkComboBox(
            self.frame_trans,
            values=[
                "local-ctranslate2 (Fastest GPU, Offline)",
                "local-qwen (Context-Aware LLM, High Quality GPU)",
                "local-transformer (Standard NLLB)",
                "google (Fast, Online)"
            ],
            state="readonly"
        )
        self.combo_translator.set("local-ctranslate2 (Fastest GPU, Offline)")
        self.combo_translator.grid(row=4, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # --- 3. Utilities & Tools ---
        self.frame_tools = ctk.CTkFrame(self.col1)
        self.frame_tools.pack(fill="x", padx=4, pady=(4, 8))
        self.frame_tools.grid_columnconfigure(0, weight=1)
        self.frame_tools.grid_columnconfigure(1, weight=1)
        self.frame_tools.grid_columnconfigure(2, weight=1)

        self.lbl_tools_hdr = ctk.CTkLabel(self.frame_tools, text="🛠️ Utilities", font=("Arial", 12, "bold"))
        self.lbl_tools_hdr.grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w")

        self.btn_sub_tool = ctk.CTkButton(
            self.frame_tools,
            text="📄 .SRT Tool",
            fg_color="#2E7D32",
            hover_color="#1B5E20",
            command=self.open_subtitle_tool
        )
        self.btn_sub_tool.grid(row=1, column=0, padx=(10, 3), pady=(2, 10), sticky="ew")

        self.btn_yt_tool = ctk.CTkButton(
            self.frame_tools,
            text="⬇️ YouTube DL",
            fg_color="#C62828",
            hover_color="#8E0000",
            command=self.open_youtube_downloader
        )
        self.btn_yt_tool.grid(row=1, column=1, padx=(3, 3), pady=(2, 10), sticky="ew")

        self.btn_logs = ctk.CTkButton(
            self.frame_tools,
            text="📁 Logs Folder",
            fg_color="#37474F",
            hover_color="#263238",
            command=self.open_logs_folder
        )
        self.btn_logs.grid(row=1, column=2, padx=(3, 10), pady=(2, 10), sticky="ew")

        # =========================================================
        # COLUMN 2: Voice Synthesis & AI Engine
        # =========================================================
        self.col2 = ctk.CTkScrollableFrame(self, label_text="🎤 Voice & AI Engine")
        self.col2.grid(row=0, column=1, padx=(6, 6), pady=12, sticky="nsew")
        self.col2.grid_columnconfigure(1, weight=1)

        # --- Voice & TTS ---
        self.frame_voice = ctk.CTkFrame(self.col2)
        self.frame_voice.pack(fill="x", padx=4, pady=(4, 8))
        self.frame_voice.grid_columnconfigure(1, weight=1)

        self.lbl_voice_hdr = ctk.CTkLabel(self.frame_voice, text="🎤 Speech Synthesis (TTS)", font=("Arial", 12, "bold"))
        self.lbl_voice_hdr.grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w")

        # TTS Provider
        self.lbl_tts = ctk.CTkLabel(self.frame_voice, text="TTS Provider:")
        self.lbl_tts.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combo_tts = ctk.CTkComboBox(
            self.frame_voice,
            values=["omnivoice (Voice Clone, Natural)", "edge-tts (Natural, Fast)", "mms (Basic, Local)"],
            state="readonly"
        )
        self.combo_tts.set("omnivoice (Voice Clone, Natural)")
        self.combo_tts.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Target Voice Lang
        self.lbl_tts_lang = ctk.CTkLabel(self.frame_voice, text="Target Voice:")
        self.lbl_tts_lang.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.combo_tts_lang = ctk.CTkComboBox(
            self.frame_voice,
            values=["Thai (th)", "English (en)"],
            state="readonly"
        )
        self.combo_tts_lang.set("Thai (th)")
        self.combo_tts_lang.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Male Pitch
        self.lbl_male_pitch = ctk.CTkLabel(self.frame_voice, text="Male Pitch:")
        self.lbl_male_pitch.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.slider_male = ctk.CTkSlider(self.frame_voice, from_=-10, to=10, number_of_steps=20)
        self.slider_male.set(0)
        self.slider_male.grid(row=3, column=1, padx=(6, 2), pady=5, sticky="ew")
        self.lbl_male_val = ctk.CTkLabel(self.frame_voice, text="0.0", width=30)
        self.lbl_male_val.grid(row=3, column=2, padx=(0, 8), pady=5)
        self.slider_male.configure(command=lambda val: self.lbl_male_val.configure(text=f"{val:.1f}"))

        # Female Pitch
        self.lbl_fem_pitch = ctk.CTkLabel(self.frame_voice, text="Female Pitch:")
        self.lbl_fem_pitch.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.slider_fem = ctk.CTkSlider(self.frame_voice, from_=-10, to=10, number_of_steps=20)
        self.slider_fem.set(0)
        self.slider_fem.grid(row=4, column=1, padx=(6, 2), pady=5, sticky="ew")
        self.lbl_fem_val = ctk.CTkLabel(self.frame_voice, text="0.0", width=30)
        self.lbl_fem_val.grid(row=4, column=2, padx=(0, 8), pady=5)
        self.slider_fem.configure(command=lambda val: self.lbl_fem_val.configure(text=f"{val:.1f}"))

        # Clone Mode
        self.lbl_clone_mode = ctk.CTkLabel(self.frame_voice, text="Clone Mode:")
        self.lbl_clone_mode.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.combo_clone_mode = ctk.CTkComboBox(
            self.frame_voice,
            values=[
                "Tone Only (Voice Tone, No Foreign Accent)",
                "Full Clone (Voice + Original Accent)",
                "Disabled (Standard TTS)"
            ],
            state="readonly"
        )
        self.combo_clone_mode.set("Tone Only (Voice Tone, No Foreign Accent)")
        self.combo_clone_mode.grid(row=5, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Reuse Speaker Voice Option
        self.chk_reuse_speaker = ctk.CTkCheckBox(self.frame_voice, text="Consistent Speaker (Reuse Prompt)")
        self.chk_reuse_speaker.select()
        self.chk_reuse_speaker.grid(row=6, column=0, columnspan=3, padx=10, pady=6, sticky="w")

        # Dual Audio Option
        self.chk_dual_audio = ctk.CTkCheckBox(self.frame_voice, text="Dual Audio (Keep Original + Add Dubbed)")
        self.chk_dual_audio.select()
        self.chk_dual_audio.grid(row=7, column=0, columnspan=3, padx=10, pady=(2, 6), sticky="w")

        # --- Gender & Multitask ---
        self.frame_perf = ctk.CTkFrame(self.col2)
        self.frame_perf.pack(fill="x", padx=4, pady=4)
        self.frame_perf.grid_columnconfigure(1, weight=1)

        self.lbl_perf_hdr = ctk.CTkLabel(self.frame_perf, text="👤 Detection & Multitask", font=("Arial", 12, "bold"))
        self.lbl_perf_hdr.grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w")

        # Gender Method
        self.lbl_gender = ctk.CTkLabel(self.frame_perf, text="Gender Method:")
        self.lbl_gender.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combo_gender = ctk.CTkComboBox(
            self.frame_perf,
            values=["audio (Fast, Pitch)", "visual (Accurate, Face)", "hybrid (Best, Slow)"],
            state="readonly"
        )
        self.combo_gender.set("audio (Fast, Pitch)")
        self.combo_gender.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Audio Model
        self.lbl_audio_model = ctk.CTkLabel(self.frame_perf, text="Audio Model:")
        self.lbl_audio_model.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.combo_audio_model = ctk.CTkComboBox(
            self.frame_perf,
            values=[
                "ml-robust (audeering, Multi-lingual AI)",
                "ml-librispeech (wav2vec2, Legacy)",
                "pitch (Hz Frequency, Fast)"
            ],
            state="readonly"
        )
        self.combo_audio_model.set("ml-robust (audeering, Multi-lingual AI)")
        self.combo_audio_model.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="ew")

        # Speaker Diarization / Clustering & Majority Voting
        self.chk_speaker_clustering = ctk.CTkCheckBox(self.frame_perf, text="Lock Gender per Speaker (Clustering)")
        self.chk_speaker_clustering.select()
        self.chk_speaker_clustering.grid(row=3, column=0, columnspan=3, padx=10, pady=(4, 6), sticky="w")

        # Multitask
        self.chk_multitask = ctk.CTkCheckBox(self.frame_perf, text="Multitask (Step 5 & 6)")
        self.chk_multitask.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.slider_workers = ctk.CTkSlider(self.frame_perf, from_=1, to=16, number_of_steps=15)
        self.slider_workers.set(4)
        self.slider_workers.grid(row=4, column=1, padx=(6, 2), pady=5, sticky="ew")
        self.lbl_workers_val = ctk.CTkLabel(self.frame_perf, text="4 Tasks", width=50)
        self.lbl_workers_val.grid(row=4, column=2, padx=(0, 8), pady=5)
        self.slider_workers.configure(command=lambda val: self.lbl_workers_val.configure(text=f"{int(val)} Tasks"))

        # =========================================================
        # COLUMN 3: Monitor, Logs & Actions
        # =========================================================
        self.col3 = ctk.CTkFrame(self)
        self.col3.grid(row=0, column=2, padx=(6, 12), pady=12, sticky="nsew")
        self.col3.grid_columnconfigure(0, weight=1)
        self.col3.grid_rowconfigure(1, weight=1)  # Logs expand vertically

        # --- System Monitor ---
        self.frame_monitor = ctk.CTkFrame(self.col3)
        self.frame_monitor.grid(row=0, column=0, padx=8, pady=(8, 4), sticky="ew")
        self.frame_monitor.grid_columnconfigure(1, weight=1)

        self.lbl_mon_title = ctk.CTkLabel(self.frame_monitor, text="📊 System Resources", font=("Arial", 12, "bold"))
        self.lbl_mon_title.grid(row=0, column=0, columnspan=3, padx=10, pady=(5, 2), sticky="w")

        # CPU
        self.lbl_cpu = ctk.CTkLabel(self.frame_monitor, text="CPU Usage:")
        self.lbl_cpu.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.prog_cpu = ctk.CTkProgressBar(self.frame_monitor)
        self.prog_cpu.grid(row=1, column=1, padx=10, pady=2, sticky="ew")
        self.prog_cpu.set(0)
        self.lbl_cpu_val = ctk.CTkLabel(self.frame_monitor, text="0%", width=45)
        self.lbl_cpu_val.grid(row=1, column=2, padx=10, pady=2)

        # GPU
        self.lbl_gpu = ctk.CTkLabel(self.frame_monitor, text="GPU Usage:")
        self.lbl_gpu.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        self.prog_gpu = ctk.CTkProgressBar(self.frame_monitor)
        self.prog_gpu.grid(row=2, column=1, padx=10, pady=2, sticky="ew")
        self.prog_gpu.set(0)
        self.lbl_gpu_val = ctk.CTkLabel(self.frame_monitor, text="0%", width=45)
        self.lbl_gpu_val.grid(row=2, column=2, padx=10, pady=2)

        # Disk
        self.lbl_disk = ctk.CTkLabel(self.frame_monitor, text="Disk Space:")
        self.lbl_disk.grid(row=3, column=0, padx=10, pady=2, sticky="w")
        self.prog_disk = ctk.CTkProgressBar(self.frame_monitor)
        self.prog_disk.grid(row=3, column=1, padx=10, pady=2, sticky="ew")
        self.prog_disk.set(0)
        self.lbl_disk_val = ctk.CTkLabel(self.frame_monitor, text="0%", width=45)
        self.lbl_disk_val.grid(row=3, column=2, padx=10, pady=2)

        # Start Monitor Thread
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

        # --- Real-time Logs ---
        self.frame_logs_container = ctk.CTkFrame(self.col3)
        self.frame_logs_container.grid(row=1, column=0, padx=8, pady=4, sticky="nsew")
        self.frame_logs_container.grid_columnconfigure(0, weight=1)
        self.frame_logs_container.grid_rowconfigure(1, weight=1)

        self.lbl_logs_hdr = ctk.CTkLabel(self.frame_logs_container, text="📋 Execution Logs", font=("Arial", 12, "bold"))
        self.lbl_logs_hdr.grid(row=0, column=0, padx=10, pady=(5, 2), sticky="w")

        self.textbox_log = ctk.CTkTextbox(self.frame_logs_container)
        self.textbox_log.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")

        # --- Actions & Progress Card ---
        self.frame_actions = ctk.CTkFrame(self.col3)
        self.frame_actions.grid(row=2, column=0, padx=8, pady=(4, 8), sticky="ew")
        self.frame_actions.grid_columnconfigure(0, weight=1)
        self.frame_actions.grid_columnconfigure(1, weight=1)

        # Progressbar
        self.progressbar = ctk.CTkProgressBar(self.frame_actions)
        self.progressbar.grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 2), sticky="ew")
        self.progressbar.set(0)

        # Status text
        self.lbl_status = ctk.CTkLabel(self.frame_actions, text="Ready", anchor="w")
        self.lbl_status.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 6), sticky="w")

        # Action Buttons: STOP and START DUBBING
        self.btn_stop = ctk.CTkButton(
            self.frame_actions,
            text="STOP",
            font=("Arial", 15, "bold"),
            height=40,
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            command=self.stop_process,
            state="disabled"
        )
        self.btn_stop.grid(row=2, column=0, padx=(10, 4), pady=(2, 8), sticky="ew")

        self.btn_start = ctk.CTkButton(
            self.frame_actions,
            text="START DUBBING",
            font=("Arial", 15, "bold"),
            height=40,
            command=self.start_thread
        )
        self.btn_start.grid(row=2, column=1, padx=(4, 10), pady=(2, 8), sticky="ew")

        # Action Button: RESET SETTINGS (NEW TASK)
        self.btn_reset = ctk.CTkButton(
            self.frame_actions,
            text="🔄 Reset Settings (New Task)",
            font=("Arial", 12, "bold"),
            height=32,
            fg_color="#455A64",
            hover_color="#37474F",
            command=self.reset_all_settings
        )
        self.btn_reset.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="ew")

        self.stop_event = None
        self.sub_window = None
        self.yt_window = None

    def open_subtitle_tool(self):
        if hasattr(self, 'sub_window') and self.sub_window is not None and self.sub_window.winfo_exists():
            self.sub_window.focus()
        else:
            self.sub_window = SubtitleTranslatorWindow(self)

    def open_youtube_downloader(self):
        if hasattr(self, 'yt_window') and self.yt_window is not None and self.yt_window.winfo_exists():
            self.yt_window.focus()
        else:
            self.yt_window = YouTubeDownloaderWindow(self)

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

    def open_logs_folder(self):
        logs_dir = os.path.abspath(Config.LOGS_DIR)
        os.makedirs(logs_dir, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(logs_dir)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", logs_dir])
            else:
                import subprocess
                subprocess.run(["xdg-open", logs_dir])
        except Exception as e:
            self.log(f"⚠️ Could not open logs directory: {e}")

    def log(self, message):
        gui_logger.info(message)
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
        temperature = 0.3

        # Extract selections
        stt_choice = self.combo_stt.get()
        translator_choice = self.combo_translator.get()
        source_lang_choice = self.combo_source_lang.get()
        target_lang_choice = self.combo_target_lang.get()
        tts_choice = self.combo_tts.get()
        tts_lang_choice = self.combo_tts_lang.get()
        gender_choice = self.combo_gender.get()
        audio_model_choice = self.combo_audio_model.get()

        # Map friendly names to codes
        stt_map = {
            "Kotoba-Whisper (Japanese SOTA, Ultra Accurate)": "kotoba-whisper",
            "faster-whisper (Large-v2, Universal)": "faster-whisper"
        }
        stt_engine = stt_map.get(stt_choice, "kotoba-whisper")

        translator_map = {
            "local-ctranslate2 (Fastest GPU, Offline)": "local-ctranslate2",
            "local-qwen (Context-Aware LLM, High Quality GPU)": "local-qwen",
            "local-transformer (Standard NLLB)": "local-transformer",
            "local-transformer (Fast, Local)": "local-transformer",
            "google (Fast, Online)": "google"
        }
        source_lang_map = {
            "Auto Detect (auto)": "auto",
            "Japanese (ja)": "ja",
            "Chinese (zh)": "zh",
            "Korean (ko)": "ko",
            "English (en)": "en",
            "Thai (th)": "th"
        }
        target_lang_map = {
            "Thai (th)": "th",
            "English (en)": "en"
        }
        tts_map = {
            "omnivoice (Voice Clone, Natural)": "omnivoice",
            "edge-tts (Natural, Fast)": "edge-tts",
            "mms (Basic, Local)": "mms"
        }
        tts_lang_map = {
            "Thai (th)": "th",
            "English (en)": "en"
        }
        gender_map = {
            "audio (Fast, Pitch)": "audio",
            "visual (Accurate, Face)": "visual",
            "hybrid (Best, Slow)": "hybrid"
        }
        audio_model_map = {
            "ml-robust (audeering, Multi-lingual AI)": "ml-robust",
            "ml-librispeech (wav2vec2, Legacy)": "ml-librispeech",
            "pitch (Hz Frequency, Fast)": "pitch",
            "ml (AI, Accurate)": "ml-robust",
            "pitch (Hz, Fast)": "pitch"
        }

        clone_mode_choice = self.combo_clone_mode.get()
        clone_mode_map = {
            "Tone Only (Voice Tone, No Foreign Accent)": "timbre",
            "Full Clone (Voice + Original Accent)": "full",
            "Disabled (Standard TTS)": "disabled",
            "Full Clone (Voice + Accent)": "full",
            "Timbre Only (Voice Only, No Accent)": "timbre"
        }

        # Multitask, Clone, Reuse Speaker Voice, and Speaker Clustering selections
        enable_multitask = bool(self.chk_multitask.get())
        max_workers = int(self.slider_workers.get())
        clone_mode = clone_mode_map.get(clone_mode_choice, "full")
        use_clone = (clone_mode != "disabled")
        reuse_speaker_voice = bool(self.chk_reuse_speaker.get())
        dual_audio = bool(self.chk_dual_audio.get())
        speaker_clustering = bool(self.chk_speaker_clustering.get())

        translator_provider = translator_map.get(translator_choice, "ollama")
        source_lang = source_lang_map.get(source_lang_choice, "auto")
        target_lang = target_lang_map.get(target_lang_choice, "th")
        tts_provider = tts_map.get(tts_choice, "omnivoice")
        tts_lang = tts_lang_map.get(tts_lang_choice, "th")
        gender_method = gender_map.get(gender_choice, "audio")
        audio_model = audio_model_map.get(audio_model_choice, "ml-robust")

        self.btn_start.configure(state="disabled", text="Running...")
        self.btn_stop.configure(state="normal")
        self.progressbar.set(0)
        self.textbox_log.delete("1.0", "end")

        self.stop_event = threading.Event()

        thread = threading.Thread(
            target=self.run_process,
            args=(input_file, output_file, male_pitch, female_pitch, temperature,
                  translator_provider, tts_provider, gender_method, audio_model,
                  source_lang, target_lang, tts_lang, enable_multitask, max_workers, use_clone, clone_mode, reuse_speaker_voice,
                  stt_engine, dual_audio, speaker_clustering)
        )
        thread.start()

    def stop_process(self):
        if self.stop_event:
            self.stop_event.set()
            self.log("Stopping... please wait for current step to finish.")
            self.btn_stop.configure(state="disabled")

    def run_process(self, input_file, output_file, male_pitch, female_pitch, temperature,
                    translator_provider, tts_provider, gender_method, audio_model,
                    source_lang, target_lang, tts_lang, enable_multitask, max_workers, use_clone, clone_mode, reuse_speaker_voice,
                    stt_engine="kotoba-whisper", dual_audio=True, speaker_clustering=True):
        # Temporarily override Config settings with GUI selections
        _apply_torchaudio_patch()
        from config import Config
        from main import run_pipeline
        original_stt = getattr(Config, "STT_ENGINE", "kotoba-whisper")
        original_translator = Config.TRANSLATION_PROVIDER
        original_tts = Config.TTS_PROVIDER
        original_gender = Config.GENDER_DETECTION_METHOD
        original_audio_model = Config.AUDIO_GENDER_MODEL
        original_multitask = Config.ENABLE_MULTITASK
        original_workers = Config.MAX_WORKERS
        original_clone = Config.OMNIVOICE_USE_CLONE
        original_clone_mode = Config.OMNIVOICE_CLONE_MODE
        original_reuse_prompt = Config.OMNIVOICE_REUSE_PROMPT
        original_dual_audio = getattr(Config, "DUAL_AUDIO_TRACKS", True)
        original_clustering = getattr(Config, "SPEAKER_CLUSTERING_GENDER", True)

        try:
            # Apply GUI selections
            Config.STT_ENGINE = stt_engine
            Config.TRANSLATION_PROVIDER = translator_provider
            Config.TTS_PROVIDER = tts_provider
            Config.GENDER_DETECTION_METHOD = gender_method
            Config.AUDIO_GENDER_MODEL = audio_model
            Config.ENABLE_MULTITASK = enable_multitask
            Config.MAX_WORKERS = max_workers
            Config.OMNIVOICE_USE_CLONE = use_clone
            Config.OMNIVOICE_CLONE_MODE = clone_mode
            Config.OMNIVOICE_REUSE_PROMPT = reuse_speaker_voice
            Config.DUAL_AUDIO_TRACKS = dual_audio
            Config.SPEAKER_CLUSTERING_GENDER = speaker_clustering

            success = run_pipeline(
                input_file,
                output_file,
                language=target_lang,
                progress_callback=self.context_progress_wrapper,
                log_callback=self.context_log_wrapper,
                male_pitch=male_pitch,
                female_pitch=female_pitch,
                translation_temperature=temperature,
                stop_event=self.stop_event,
                source_lang=source_lang,
                target_lang=target_lang,
                tts_lang=tts_lang,
                enable_multitask=enable_multitask,
                max_workers=max_workers,
                use_clone=use_clone,
                clone_mode=clone_mode,
                reuse_speaker_voice=reuse_speaker_voice,
                stt_engine=stt_engine,
                dual_audio=dual_audio
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
            Config.STT_ENGINE = original_stt
            Config.TRANSLATION_PROVIDER = original_translator
            Config.TTS_PROVIDER = original_tts
            Config.GENDER_DETECTION_METHOD = original_gender
            Config.AUDIO_GENDER_MODEL = original_audio_model
            Config.ENABLE_MULTITASK = original_multitask
            Config.MAX_WORKERS = original_workers
            Config.OMNIVOICE_USE_CLONE = original_clone
            Config.OMNIVOICE_CLONE_MODE = original_clone_mode
            Config.OMNIVOICE_REUSE_PROMPT = original_reuse_prompt
            Config.DUAL_AUDIO_TRACKS = original_dual_audio
            Config.SPEAKER_CLUSTERING_GENDER = original_clustering
            self.after(0, lambda: self.reset_buttons())

    def reset_buttons(self):
        self.btn_start.configure(state="normal", text="START DUBBING")
        self.btn_stop.configure(state="disabled")

    def reset_all_settings(self):
        """Resets all GUI controls, file paths, and options back to clean defaults for a new task."""
        # 1. Clear file paths
        self.entry_input.delete(0, "end")
        self.entry_output.delete(0, "end")

        # 2. Reset STT & Translation
        self.combo_stt.set("Kotoba-Whisper (Japanese SOTA, Ultra Accurate)")
        self.combo_source_lang.set("Auto Detect (auto)")
        self.combo_target_lang.set("Thai (th)")
        self.combo_translator.set("local-ctranslate2 (Fastest GPU, Offline)")

        # 3. Reset TTS & Voice
        self.combo_tts.set("omnivoice (Voice Clone, Natural)")
        self.combo_tts_lang.set("Thai (th)")
        self.slider_male.set(0)
        self.lbl_male_val.configure(text="0.0")
        self.slider_fem.set(0)
        self.lbl_fem_val.configure(text="0.0")
        self.combo_clone_mode.set("Tone Only (Voice Tone, No Foreign Accent)")
        self.chk_reuse_speaker.select()
        self.chk_dual_audio.select()

        # 4. Reset Gender & Performance
        self.combo_gender.set("audio (Fast, Pitch)")
        self.combo_audio_model.set("ml-robust (audeering, Multi-lingual AI)")
        self.chk_speaker_clustering.select()
        self.chk_multitask.deselect()
        self.slider_workers.set(4)
        self.lbl_workers_val.configure(text="4 Tasks")

        # 5. Reset Progress & Status
        self.progressbar.set(0)
        self.lbl_status.configure(text="Ready")
        self.reset_buttons()

        self.log("🔄 Reset all settings and file paths to default for new task.")

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
    import traceback
    try:
        print("Launching GUI window...", flush=True)
        app = App()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Bring window to foreground on Windows
        app.lift()
        app.attributes('-topmost', True)
        app.after_idle(app.attributes, '-topmost', False)
        app.focus_force()
        
        print("GUI is running successfully!", flush=True)
        app.mainloop()
    except Exception as e:
        err = traceback.format_exc()
        with open("gui_error.log", "w", encoding="utf-8") as f:
            f.write(err)
        print(f"\n❌ Error launching GUI: {e}\n{err}", flush=True)
        input("Press Enter to exit...")
