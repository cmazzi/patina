#!/usr/bin/env python3
"""
Tk front end for patina.py.

A thin wrapper: it builds a command line and runs patina.py as a
subprocess, streaming its output into the log window. Everything the tool can
do is still available from the terminal, and the command actually launched is
printed in the log so it can be copied and reused.

    python3 patina_gui.py

MIT licence. Copyright (c) 2026 Carlo Mazzi. See LICENSE.
"""

import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from patina import PRESETS, Params, AUDIO_EXT     # noqa: E402

SCRIPT = HERE / "patina.py"

# Which parameters go in which box, and how they are labelled. The order here
# is the order on screen; anything in Params but not listed is left at default.
GROUPS = [
    ("Non-linearity", [
        ("drive",         "Drive (0-1)"),
        ("bias",          "Bias (0-1)"),
        ("mix",           "Mix (%)"),
        ("oversample",    "Oversampling"),
    ]),
    ("Vinyl", [
        ("bass_mono_hz",  "Bass to mono below (Hz)"),
        ("crosstalk_db",  "Crosstalk (dB)"),
        ("wow_pct",       "Wow (%)"),
        ("flutter_pct",   "Flutter (%)"),
        ("hf_rolloff_db", "HF roll-off (dB)"),
        ("tilt_db",       "Tracking tilt (dB)"),
        ("rumble_db",     "Rumble (dB)"),
        ("noise_db",      "Surface hiss (dB)"),
        ("click_rate",    "Clicks per second"),
        ("click_db",      "Click level (dB)"),
        ("tick_db",       "Periodic pop (dB)"),
        ("rpm",           "Platter speed (rpm)"),
    ]),
    ("Valve stage (mode 'both')", [
        ("tube_drive",    "Tube drive (0-1)"),
        ("tube_bias",     "Tube bias (0-1)"),
    ]),
    ("Output", [
        ("headroom_db",   "Headroom (dB)"),
    ]),
]

INT_FIELDS = {"oversample"}
HINT = "-999 = off"


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.proc = None
        self.stopping = False
        self.q = queue.Queue()
        self.vars = {}

        self._build_paths()
        self._build_params()
        self._build_options()
        self._build_actions()
        self._build_log()

        self.apply_preset()
        self.after(100, self._drain)

    # ---------------------------------------------------------------- layout

    def _build_paths(self):
        box = ttk.LabelFrame(self, text="Files", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)

        self.src = tk.StringVar()
        self.dst = tk.StringVar()

        ttk.Label(box, text="Source").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.src).grid(row=0, column=1,
                                                   sticky="ew", padx=6)
        ttk.Button(box, text="Folder…", width=8,
                   command=self.pick_src_dir).grid(row=0, column=2)
        ttk.Button(box, text="File…", width=7,
                   command=self.pick_src_file).grid(row=0, column=3, padx=(4, 0))

        ttk.Label(box, text="Destination").grid(row=1, column=0, sticky="w",
                                                pady=(6, 0))
        ttk.Entry(box, textvariable=self.dst).grid(row=1, column=1, sticky="ew",
                                                   padx=6, pady=(6, 0))
        ttk.Button(box, text="Folder…", width=8,
                   command=self.pick_dst).grid(row=1, column=2, pady=(6, 0))

        self.src.trace_add("write", lambda *_: self.suggest_dst())

    def _build_params(self):
        box = ttk.LabelFrame(self, text="Coloration", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        box.columnconfigure(0, weight=1)

        head = ttk.Frame(box)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.preset = tk.StringVar(value="tube")
        self.mode = tk.StringVar(value="tube")

        ttk.Label(head, text="Preset").pack(side="left")
        cb = ttk.Combobox(head, textvariable=self.preset, width=15,
                          state="readonly", values=list(PRESETS))
        cb.pack(side="left", padx=(6, 16))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.apply_preset())

        ttk.Label(head, text="Mode").pack(side="left")
        ttk.Combobox(head, textvariable=self.mode, width=8, state="readonly",
                     values=["tube", "vinyl", "both"]).pack(side="left",
                                                            padx=(6, 16))
        ttk.Button(head, text="Reset to preset",
                   command=self.apply_preset).pack(side="left")

        cols = ttk.Frame(box)
        cols.grid(row=1, column=0, sticky="ew")
        for i in range(len(GROUPS)):
            cols.columnconfigure(i, weight=1)

        defaults = asdict(Params())
        for col, (title, fields) in enumerate(GROUPS):
            g = ttk.LabelFrame(cols, text=title, padding=6)
            g.grid(row=0, column=col, sticky="nsew", padx=(0, 8))
            for row, (name, label) in enumerate(fields):
                ttk.Label(g, text=label).grid(row=row, column=0, sticky="w")
                v = tk.StringVar(value=str(defaults[name]))
                self.vars[name] = v
                ttk.Entry(g, textvariable=v, width=8).grid(row=row, column=1,
                                                           sticky="e", padx=(6, 0))
            if title == "Vinyl":
                ttk.Label(g, text=HINT, foreground="grey").grid(
                    row=len(fields), column=0, columnspan=2, sticky="w",
                    pady=(4, 0))

    def _build_options(self):
        box = ttk.LabelFrame(self, text="Options", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.recursive = tk.BooleanVar(value=False)
        self.force = tk.BooleanVar(value=False)
        self.match_rms = tk.BooleanVar(value=True)
        self.dither = tk.BooleanVar(value=True)
        self.jobs = tk.StringVar(value="")

        ttk.Checkbutton(box, text="Recursive",
                        variable=self.recursive).pack(side="left")
        ttk.Checkbutton(box, text="Overwrite existing",
                        variable=self.force).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(box, text="Match RMS",
                        variable=self.match_rms).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(box, text="Dither 16-bit",
                        variable=self.dither).pack(side="left", padx=(12, 0))
        ttk.Label(box, text="Jobs").pack(side="left", padx=(18, 4))
        ttk.Entry(box, textvariable=self.jobs, width=5).pack(side="left")
        ttk.Label(box, text="(empty = automatic)",
                  foreground="grey").pack(side="left", padx=(6, 0))

    def _build_actions(self):
        bar = ttk.Frame(self)
        bar.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bar.columnconfigure(4, weight=1)

        self.b_run = ttk.Button(bar, text="Process", command=self.run)
        self.b_run.grid(row=0, column=0)
        self.b_dry = ttk.Button(bar, text="Dry run",
                                command=lambda: self.run(dry=True))
        self.b_dry.grid(row=0, column=1, padx=(6, 0))
        self.b_ana = ttk.Button(bar, text="Analyze",
                                command=lambda: self.run(analyze=True))
        self.b_ana.grid(row=0, column=2, padx=(6, 0))
        self.b_stop = ttk.Button(bar, text="Stop", command=self.stop,
                                 state="disabled")
        self.b_stop.grid(row=0, column=3, padx=(6, 0))

        self.progress = ttk.Progressbar(bar, mode="determinate")
        self.progress.grid(row=0, column=4, sticky="ew", padx=(12, 0))

    def _build_log(self):
        box = ttk.Frame(self)
        box.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        self.log = tk.Text(box, height=14, wrap="none", font=("Menlo", 11))
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(box, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set, state="disabled")

    # ------------------------------------------------------------- behaviour

    def pick_src_dir(self):
        d = filedialog.askdirectory(title="Source folder")
        if d:
            self.src.set(d)

    def pick_src_file(self):
        # On macOS a filetypes entry with several patterns is unreliable, so
        # the extensions are listed one per entry.
        types = [(e.upper().lstrip("."), f"*{e}") for e in sorted(AUDIO_EXT)]
        f = filedialog.askopenfilename(title="Source file", filetypes=types)
        if f:
            self.src.set(f)

    def pick_dst(self):
        d = filedialog.askdirectory(title="Destination folder")
        if d:
            self.dst.set(d)

    def suggest_dst(self):
        """Propose '<source> [preset]' next to the source, if left empty."""
        s = self.src.get().strip()
        if not s or self.dst.get().strip():
            return
        p = Path(s)
        base = p if p.is_dir() else p.parent
        self.dst.set(str(base.parent / f"{base.name} [{self.preset.get()}]"))

    def apply_preset(self):
        """Load the preset values into the fields, leaving the rest at default."""
        values = asdict(Params())
        values.update(PRESETS[self.preset.get()])
        self.mode.set(values["mode"])
        for name, var in self.vars.items():
            var.set(str(values[name]))
        self.match_rms.set(values["match_rms"])
        self.dither.set(values["dither"])

    def build_cmd(self, dry=False, analyze=False):
        cmd = [sys.executable, "-u", str(SCRIPT)]
        if analyze:
            cmd.append("--analyze")
        else:
            src = self.src.get().strip()
            dst = self.dst.get().strip()
            if not src:
                raise ValueError("Choose a source file or folder.")
            if not Path(src).exists():
                raise ValueError(f"Source not found:\n{src}")
            if not dst:
                raise ValueError("Choose a destination folder.")
            cmd += [src, "-o", dst]
            if self.recursive.get():
                cmd.append("--recursive")
            if self.force.get():
                cmd.append("--force")
            if dry:
                cmd.append("--dry-run")
            if self.jobs.get().strip():
                cmd += ["--jobs", self.jobs.get().strip()]

        cmd += ["--mode", self.mode.get()]
        for name, var in self.vars.items():
            text = var.get().strip()
            if not text:
                raise ValueError(f"Empty value for '{name}'.")
            try:
                int(text) if name in INT_FIELDS else float(text)
            except ValueError:
                raise ValueError(f"'{text}' is not a number ({name}).")
            cmd += ["--" + name.replace("_", "-"), text]
        cmd.append("--match-rms" if self.match_rms.get() else "--no-match-rms")
        cmd.append("--dither" if self.dither.get() else "--no-dither")
        return cmd

    def run(self, dry=False, analyze=False):
        if self.proc is not None:
            return
        try:
            cmd = self.build_cmd(dry=dry, analyze=analyze)
        except ValueError as e:
            messagebox.showwarning("patina", str(e))
            return

        self.log_clear()
        # Quoted so the line can be pasted straight into a shell.
        self.write("$ " + " ".join(
            f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
        self.progress.configure(value=0, maximum=100)
        self.stopping = False
        self.set_running(True)
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, cwd=str(HERE),
                # Own process group, so Stop takes the worker pool down too
                # and not just the parent.
                start_new_session=True)
        except OSError as e:
            self.q.put(("line", f"Could not start: {e}\n"))
            self.q.put(("done", None))
            return
        for line in self.proc.stdout:
            self.q.put(("line", line))
        code = self.proc.wait()
        self.q.put(("done", code))

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.stopping = True
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except OSError:
                self.proc.terminate()
            self.write("\n-- stopped --\n")

    def _drain(self):
        """Move worker output into the widgets, in the Tk thread."""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self.write(payload)
                    self._progress_from(payload)
                else:
                    self.proc = None
                    self.set_running(False)
                    if payload not in (0, None) and not self.stopping:
                        self.write(f"\n-- exit code {payload} --\n")
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _progress_from(self, line):
        """patina.py prints '[i/n] status name'; use it to drive the bar."""
        if not line.startswith("["):
            return
        head = line[1:line.find("]")] if "]" in line else ""
        if "/" not in head:
            return
        i, _, n = head.partition("/")
        try:
            self.progress.configure(maximum=int(n), value=int(i))
        except ValueError:
            pass

    def set_running(self, running):
        state = "disabled" if running else "normal"
        for b in (self.b_run, self.b_dry, self.b_ana):
            b.configure(state=state)
        self.b_stop.configure(state="normal" if running else "disabled")

    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


def main():
    root = tk.Tk()
    root.title("patina")
    root.minsize(900, 640)
    app = App(root)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (app.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
