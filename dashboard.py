"""Desktop controls and capture monitor."""

import json
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from farmer import validate_config

ROOT = Path(__file__).resolve().parent


class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clash Farmer")
        self.geometry("1180x860")
        self.minsize(900, 650)
        self.process = None
        self.events = queue.Queue()
        self.started = None
        self.closing = False
        self.frame_count = 0
        self.battles = 0
        self.gained = {"gold": 0, "elixir": 0, "dark_elixir": 0}
        self.gained_text = tk.StringVar(value="Session loot: no results recorded")
        self.status = tk.StringVar(value="Stopped")
        self.stats = tk.StringVar(value="Uptime 00:00:00 · Frames 0 · Completed attacks 0")
        self.loot = tk.StringVar(value="Available loot: —")
        self.readiness = tk.StringVar()
        self.plan_status = tk.StringVar(value="Visual plan: start monitoring on a scouting screen.")
        self.training_status = tk.StringVar(value="Training: no run recorded")
        self.activity_action = tk.StringVar(value="Idle - game controls inactive")
        self.activity_evidence = tk.StringVar(
            value="Open a saved base to see model decisions without capture."
        )
        self.activity_next = tk.StringVar(value="Next: choose Explain saved base")
        self.activity_mode = tk.StringVar(value="Stopped")
        self.last_training_check = 0

        header = ttk.Frame(self, padding=16)
        header.pack(fill="x")
        ttk.Label(header, text="Clash Farmer", font=("Segoe UI", 22, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status, font=("Segoe UI", 12)).pack(side="right")
        controls = ttk.Frame(self, padding=(16, 0, 16, 10))
        controls.pack(fill="x")
        self.monitor_button = ttk.Button(
            controls, text="Start monitoring", command=lambda: self.start(False)
        )
        self.monitor_button.pack(side="left")
        self.attack_button = ttk.Button(
            controls, text="Run one configured attack", command=lambda: self.start(True)
        )
        self.attack_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Button(controls, text="Vision samples / analyze", command=self.open_vision).pack(
            side="left", padx=8
        )
        ttk.Label(controls, textvariable=self.stats).pack(side="right")
        sessions = ttk.Frame(self, padding=(16, 0, 16, 8))
        sessions.pack(fill="x")
        self.session_button = ttk.Button(
            sessions, text="Run continuous session", command=lambda: self.start(True, True)
        )
        self.session_button.pack(side="left")
        ttk.Label(sessions, text="Attack limit (0 = until stopped):").pack(side="left", padx=8)
        self.session_limit = tk.IntVar(value=0)
        ttk.Spinbox(sessions, from_=0, to=100000, textvariable=self.session_limit, width=7).pack(
            side="left"
        )
        ttk.Button(sessions, text="Setup status", command=self.show_setup).pack(side="left", padx=8)
        self.explain_button = ttk.Button(
            sessions, text="Explain saved base", command=self.explain_saved
        )
        self.explain_button.pack(side="left", padx=8)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        overview = ttk.Frame(notebook, padding=10)
        settings = ttk.Frame(notebook, padding=10)
        notebook.add(overview, text="Progress")
        notebook.add(settings, text="Settings and attack steps")
        activity_tab = ttk.Frame(notebook, padding=10)
        notebook.add(activity_tab, text="Activity")
        ttk.Label(
            activity_tab, text="Observed screens, measured evidence, and planned actions."
        ).pack(anchor="w", pady=6)
        history_frame = ttk.Frame(activity_tab)
        history_frame.pack(fill="both", expand=True)
        self.activity_history = ttk.Treeview(
            history_frame, columns=("time", "mode", "action", "evidence", "next"), show="headings"
        )
        for key, label, width in [
            ("time", "Time", 75),
            ("mode", "Mode", 120),
            ("action", "Action", 220),
            ("evidence", "Evidence / reason", 390),
            ("next", "Next action", 220),
        ]:
            self.activity_history.heading(key, text=label)
            self.activity_history.column(key, width=width, stretch=True)
        self.activity_history.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            history_frame, orient="vertical", command=self.activity_history.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.activity_history.configure(yscrollcommand=scrollbar.set)
        horizontal = ttk.Scrollbar(
            activity_tab, orient="horizontal", command=self.activity_history.xview
        )
        horizontal.pack(fill="x")
        self.activity_history.configure(xscrollcommand=horizontal.set)
        explanation = ttk.LabelFrame(overview, text="Current activity", padding=8)
        explanation.pack(fill="x", pady=(0, 8))
        ttk.Label(explanation, textvariable=self.activity_mode).pack(anchor="w")
        ttk.Label(
            explanation, textvariable=self.activity_action, font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")
        ttk.Label(explanation, textvariable=self.activity_evidence, wraplength=1050).pack(
            anchor="w"
        )
        ttk.Label(explanation, textvariable=self.activity_next, wraplength=1050).pack(anchor="w")
        ttk.Label(overview, textvariable=self.loot, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(overview, textvariable=self.gained_text).pack(anchor="w")
        ttk.Label(overview, textvariable=self.plan_status, wraplength=1050).pack(anchor="w")
        ttk.Label(overview, textvariable=self.training_status, wraplength=1050).pack(anchor="w")
        self.preview = ttk.Label(
            overview, text="Start monitoring to see the game window.", anchor="center"
        )
        self.preview.pack(fill="both", expand=True, pady=8)
        ttk.Label(overview, textvariable=self.readiness, wraplength=1050, justify="left").pack(
            anchor="w", pady=5
        )
        self.log = ScrolledText(overview, height=8, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="x")

        ttk.Label(
            settings,
            text="Edit loot thresholds, screen calibration, and attack_steps below. Save while stopped.",
        ).pack(anchor="w")
        self.editor = ScrolledText(settings, font=("Consolas", 11), undo=True)
        self.editor.pack(fill="both", expand=True, pady=8)
        self.save_button = ttk.Button(settings, text="Save settings", command=self.save)
        self.save_button.pack(anchor="w")
        ttk.Label(
            settings,
            text='Attack steps: {"action":"click", "point":[x,y], "count":10, "interval":0.2} or {"action":"wait", "seconds":5}',
            wraplength=1000,
        ).pack(anchor="w", pady=8)
        self.editor.insert("1.0", (ROOT / "config.json").read_text())
        self.refresh_readiness()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(150, self.poll)

    def open_vision(self):
        from vision_editor import VisionEditor

        VisionEditor(self, ROOT)

    def explain_saved(self):
        if self.process:
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Explain a saved scouting screenshot",
            initialdir=ROOT / "data" / "calibration",
            filetypes=[("Screenshots", "*.png *.jpg *.jpeg")],
        )
        if path:
            self.start(False, saved_image=path)

    def show_activity(self, event):
        self.activity_mode.set(event.get("mode", "Controller"))
        self.activity_action.set(event.get("action", ""))
        evidence = event.get("evidence", "")
        if event.get("spawn") is not None:
            evidence += f" | Deployment point: {event['spawn']}"
        self.activity_evidence.set(evidence)
        self.activity_next.set("Next: " + event.get("next_action", ""))
        row = self.activity_history.insert(
            "",
            "end",
            values=(
                time.strftime("%H:%M:%S"),
                event.get("mode", "Controller"),
                event.get("action", ""),
                evidence,
                event.get("next_action", ""),
            ),
        )
        rows = self.activity_history.get_children()
        if len(rows) > 300:
            self.activity_history.delete(*rows[: len(rows) - 300])
        self.activity_history.see(row)

    def show_setup(self):
        from calibration import readiness

        report = readiness(json.loads((ROOT / "config.json").read_text()), ROOT, session=True)
        messagebox.showinfo(
            "Setup status",
            "\n".join(report["missing"])
            or "Files configured. Supervised gameplay validation still required.",
            parent=self,
        )

    def refresh_readiness(self):
        config = json.loads((ROOT / "config.json").read_text())
        missing = []
        if config.get("vision", {}).get("geometry_validated") is not True:
            missing.append("wall/deployment geometry validation")
        for name in ("home", "attack_menu", "army_confirmation", "scouting", "battle", "results"):
            if not config.get("templates", {}).get(name):
                missing.append(name + " template")
        for name in ("attack", "find_match", "confirm_attack", "next", "return_home"):
            if name not in config.get("buttons", {}):
                missing.append(name + " button")
        if config.get("attack_mode") == "visual":
            directory = ROOT / config["vision"]["template_dir"]
            backend = config["vision"].get("backend")
            if backend in ("yolo", "hybrid"):
                if not (ROOT / config["vision"]["model_path"]).is_file():
                    missing.append("trained collector model")
            if backend != "yolo":
                for kind in ("gold_mine", "elixir_collector", "dark_drill", "wall", "rubble"):
                    if backend == "hybrid" and kind not in ("wall", "rubble"):
                        continue
                    if not list((directory / kind).glob("*.png")):
                        missing.append(kind + " vision samples")
            for name in ("surrender",):
                if name not in config["templates"]:
                    missing.append(name + " template")
            for name in ("end_battle", "confirm_surrender"):
                if name not in config["buttons"]:
                    missing.append(name + " button")
        elif not config.get("attack_steps"):
            missing.append("attack steps")
        self.readiness.set(
            "Needs calibration: " + ", ".join(missing)
            if missing
            else "Configured attack available; gameplay still requires testing."
        )
        self.readiness.set(
            self.readiness.get()
            + "\nVisual routes remain estimates. UI files alone do not verify wall coverage or unattended gameplay."
        )

    def save(self):
        if self.process:
            return
        try:
            config = json.loads(self.editor.get("1.0", "end"))
            if not isinstance(config, dict):
                raise ValueError("Settings must be a JSON object")
            for key in (
                "window_title",
                "capture_size",
                "tesseract_cmd",
                "minimum_loot",
                "templates",
                "buttons",
                "attack_steps",
                "loot_regions",
            ):
                if key not in config:
                    raise ValueError("Missing setting: " + key)
            temp = ROOT / "config.tmp.json"
            temp.write_text(json.dumps(config, indent=2) + "\n")
            temp.replace(ROOT / "config.json")
            self.refresh_readiness()
            self.add_log("Settings saved.")
        except (ValueError, OSError, TypeError) as error:
            messagebox.showerror("Settings", str(error))

    def start(self, live, session=False, saved_image=None):
        if self.process:
            return
        if live:
            try:
                config = json.loads((ROOT / "config.json").read_text())
                validate_config(config)
                if session:
                    from session_runner import validate_session

                    config.setdefault("session", {})["max_battles"] = self.session_limit.get()
                    validate_session(config)
            except (ValueError, KeyError, TypeError, tk.TclError) as error:
                messagebox.showerror("Calibration needed", str(error))
                return
        (ROOT / ".runtime").mkdir(exist_ok=True)
        self.stop_file = ROOT / ".runtime" / (uuid.uuid4().hex + ".stop")
        command = [
            sys.executable,
            "-u",
            str(ROOT / "dashboard_worker.py"),
            "--stop-file",
            str(self.stop_file),
        ]
        if saved_image:
            command = [
                sys.executable,
                "-u",
                str(ROOT / "explain_saved.py"),
                "--image",
                str(saved_image),
                "--stop-file",
                str(self.stop_file),
            ]
        if live:
            command.append("--live")
        if session:
            command.extend(["--session", "--max-battles", str(self.session_limit.get())])
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError as error:
            messagebox.showerror("Cannot start", str(error))
            return
        self.started = time.monotonic()
        self.frame_count = 0
        if live:
            self.battles = 0
            self.gained = {"gold": 0, "elixir": 0, "dark_elixir": 0}
            self.gained_text.set("Session loot: no results recorded")
        self.status.set("Starting attack — focus game" if live else "Monitoring")
        if saved_image:
            self.status.set("Analyzing saved base")
        self.activity_mode.set(
            "Saved-image preview" if saved_image else "Live attack" if live else "Monitoring"
        )
        self.activity_action.set("Starting worker")
        self.activity_evidence.set(
            "Saved files only; game controls inactive"
            if saved_image
            else "Waiting for worker updates"
        )
        self.activity_next.set("Next: load models")
        for button in (
            self.monitor_button,
            self.attack_button,
            self.session_button,
            self.save_button,
            self.explain_button,
        ):
            button.config(state="disabled")
        self.editor.config(state="disabled")
        self.stop_button.config(state="normal")
        process = self.process

        def read_output():
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    event = {"event": "log", "message": line.rstrip()}
                self.events.put(event)

        threading.Thread(target=read_output, daemon=True).start()

    def stop(self):
        process = self.process
        if process and process.poll() is None:
            self.status.set("Stopping…")
            try:
                self.stop_file.touch()
            except (OSError, ValueError):
                pass
            self.after(3000, lambda: process.terminate() if process.poll() is None else None)

    def close(self):
        self.closing = True
        if self.process:
            self.stop()
        else:
            self.destroy()

    def add_log(self, message):
        self.log.config(state="normal")
        self.log.insert("end", message + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 400:
            self.log.delete("1.0", "100.0")
        self.log.see("end")
        self.log.config(state="disabled")

    def poll(self):
        from PIL import Image, ImageTk

        if time.monotonic() - self.last_training_check > 2:
            self.last_training_check = time.monotonic()
            try:
                progress_files = list((ROOT / "runs").rglob("progress.json"))
                if progress_files:
                    latest = max(progress_files, key=lambda path: path.stat().st_mtime)
                    progress = json.loads(latest.read_text())
                    score = progress.get("metrics", {}).get("metrics/mAP50(B)")
                    metric = f" | validation mAP50 {score:.3f}" if score is not None else ""
                    self.training_status.set(
                        f"Training {latest.parent.name}: {progress['status']} | epoch {progress['epoch']}/{progress['epochs']}{metric}"
                    )
            except (OSError, ValueError, KeyError, TypeError):
                pass
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("event")
            if kind == "activity":
                self.show_activity(event)
            elif kind == "frame":
                self.frame_count = event["frames"]
                try:
                    with Image.open(ROOT / "dashboard-frame.png") as image:
                        image.thumbnail((960, 460))
                        self.photo = ImageTk.PhotoImage(image.copy())
                    self.preview.config(image=self.photo, text="")
                except OSError:
                    pass
            elif kind == "status":
                self.status.set("Monitoring · " + event["state"])
                loot = event.get("loot")
                if event["state"] == "saved-image preview":
                    self.status.set("Saved-image preview")
                plan = event.get("plan")
                self.plan_status.set(
                    (
                        f"Visual plan: {plan['collectors']} collectors, {plan['eligible']} estimated reachable. "
                        + " ".join(plan["problems"])
                    )
                    if plan
                    else "Visual plan: waiting for a scouting screen."
                )
                self.loot.set(
                    "Available loot: "
                    + (
                        " · ".join(
                            f"{key.replace('_', ' ')}: {value:,}" for key, value in loot.items()
                        )
                        if loot
                        else "— (not reading a scouting screen)"
                    )
                )
            elif kind == "completed":
                self.battles += event["battles"]
            elif kind == "loot_gained":
                for resource, value in event["loot"].items():
                    self.gained[resource] += value
                self.gained_text.set(
                    "Session loot: "
                    + " | ".join(f"{k.replace('_', ' ')}: {v:,}" for k, v in self.gained.items())
                )
            elif kind == "error":
                self.status.set("Stopped with error")
                self.activity_action.set("Stopped with error")
                self.activity_evidence.set(event["message"])
                self.activity_next.set("Next: inspect the reported error")
                self.add_log(event["message"])
            else:
                self.add_log(event.get("message", str(event)))
        if self.process:
            elapsed = int(time.monotonic() - self.started)
            self.stats.set(
                f"Uptime {elapsed // 3600:02}:{elapsed // 60 % 60:02}:{elapsed % 60:02} · Frames {self.frame_count} · Completed attacks {self.battles}"
            )
            code = self.process.poll()
            if code is not None:
                self.process.stdout.close()
                self.stop_file.unlink(missing_ok=True)
                self.process = None
                self.status.set("Stopped" if code == 0 else "Stopped with error")
                for button in (
                    self.monitor_button,
                    self.attack_button,
                    self.session_button,
                    self.save_button,
                    self.explain_button,
                ):
                    button.config(state="normal")
                self.editor.config(state="normal")
                self.stop_button.config(state="disabled")
                self.add_log(f"Worker exited ({code}).")
                if self.closing:
                    self.destroy()
                    return
        self.after(150, self.poll)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explain", type=Path, help="Analyze a saved image on startup; no game capture/input"
    )
    args = parser.parse_args()
    from window_capture import enable_dpi_awareness

    enable_dpi_awareness()
    app = Dashboard()
    if args.explain:
        app.after(100, lambda: app.start(False, saved_image=str(args.explain.resolve())))
    app.mainloop()
