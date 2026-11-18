"""Sample labeling on a frozen frame, with a detection overlay."""

import json
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from dataset import Dataset
from vision import COLLECTORS, KINDS, VisualPlanner, draw_plan


class VisionEditor(tk.Toplevel):
    def __init__(self, parent, root):
        super().__init__(parent)
        self.root = root
        self.title("Vision samples and base analysis")
        self.geometry("1140x850")
        self.image = self.anchor = self.selection = None
        self.dataset = Dataset(root / "data" / "collectors")
        self.record = None
        self.group = tk.StringVar()
        self.review_scope = tk.StringVar(value="Collectors only")
        self.detector_choice = tk.StringVar(value="Configured detector")
        self.wall_choice = tk.StringVar(value="Original wall samples")
        self.kind = tk.StringVar(value="gold_mine")
        self.status = tk.StringVar()
        toolbar = ttk.Frame(self, padding=10)
        toolbar.pack(fill="x")
        self.batch_paths = []
        self.batch_index = -1
        for text, command in [
            ("Load saved capture", self.load_latest),
            ("Open screenshot", self.open_image),
            ("Scouting batch", self.open_batch),
            ("Next frame", self.next_frame),
        ]:
            ttk.Button(toolbar, text=text, command=command).pack(side="left", padx=3)
        ttk.Combobox(
            toolbar, textvariable=self.kind, values=KINDS, state="readonly", width=19
        ).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Save selected sample", command=self.save_sample).pack(
            side="left", padx=5
        )
        ttk.Button(toolbar, text="Analyze this base", command=self.analyze).pack(
            side="left", padx=5
        )
        labeling = ttk.Frame(self, padding=(10, 0))
        labeling.pack(fill="x")
        ttk.Label(labeling, text="Base ID:").pack(side="left")
        ttk.Entry(labeling, textvariable=self.group, width=18).pack(side="left", padx=5)
        ttk.Combobox(
            labeling,
            textvariable=self.review_scope,
            values=["Collectors only", "All objects"],
            state="readonly",
            width=16,
        ).pack(side="left", padx=5)
        ttk.Button(labeling, text="Suggest boxes", command=self.suggest_labels).pack(
            side="left", padx=3
        )
        ttk.Button(labeling, text="Undo last label", command=self.undo_label).pack(
            side="left", padx=3
        )
        ttk.Button(labeling, text="Mark fully labeled", command=self.review).pack(
            side="left", padx=3
        )
        models = ttk.Frame(self, padding=(10, 5))
        models.pack(fill="x")
        ttk.Label(models, text="Offline detector:").pack(side="left")
        choices = ["Configured detector"] + [
            str(path.relative_to(root)) for path in sorted((root / "models").glob("*.pt"))
        ]
        if (root / "models" / "collectors-scouting-23.pt").exists():
            self.detector_choice.set(str(Path("models") / "collectors-scouting-23.pt"))
        if all(
            (root / "models" / name).exists()
            for name in ("walls-sparse-02.pt", "building-occupancy-01.pt")
        ):
            self.wall_choice.set("Wall network (experimental)")
        ttk.Combobox(
            models, textvariable=self.detector_choice, values=choices, state="readonly", width=40
        ).pack(side="left", padx=5)
        ttk.Combobox(
            models,
            textvariable=self.wall_choice,
            values=[
                "Original wall samples",
                "Expanded walls (experimental)",
                "Wall network (experimental)",
            ],
            state="readonly",
            width=29,
        ).pack(side="left", padx=5)
        self.canvas = tk.Canvas(
            self, width=1100, height=620, background="#202326", highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=5)
        self.canvas.bind("<ButtonPress-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.end)
        self.canvas.bind("<ButtonPress-3>", self.remove_label)
        ttk.Label(self, textvariable=self.status, wraplength=1090).pack(anchor="w", padx=10, pady=5)
        ttk.Label(
            self,
            text="Drag a tight box around one object. Right-click a saved box to remove it. Label every visible object in the selected review scope.\nKeep the same Base ID across frames of a base. Restart monitoring after saving template samples.",
            wraplength=1090,
        ).pack(anchor="w", padx=10)
        self.load_latest()

    def load_latest(self):
        self.dataset = Dataset(self.root / "data" / "collectors")
        path = self.root / "vision-frame.png"
        self.load(path if path.exists() else self.root / "preview.png")

    def open_batch(self):
        folder = self.root / "data" / "scouting-bases"
        if not (folder / "manifest.json").exists():
            self.status.set("Import scouting-bases.zip with import_scouting.py first.")
            return
        self.dataset = Dataset(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        self.batch_paths = [folder / "images" / (r["id"] + ".png") for r in manifest["frames"]]
        self.batch_index = -1
        self.next_frame()

    def next_frame(self):
        if self.batch_paths:
            self.batch_index = (self.batch_index + 1) % len(self.batch_paths)
            self.load(self.batch_paths[self.batch_index])
            self.status.set(
                f"Batch {self.batch_index + 1}/{len(self.batch_paths)} | Reserved {self.record.get('split', 'unassigned')} | Reviewed: {self.record.get('reviewed', False)}. Check every visible collector and Base ID."
            )

    def analysis_settings(self):
        config = json.loads((self.root / "config.json").read_text())
        settings = dict(config["vision"])
        if self.detector_choice.get() != "Configured detector":
            settings.update(
                backend="hybrid", model_path=str(self.root / self.detector_choice.get())
            )
            # Offline viewport only. Live capture coordinates remain unchanged.
            if self.image is not None and self.image.size == (1920, 1080):
                settings["playfield"] = [0, 0, 1920, 905]
        if self.wall_choice.get() == "Expanded walls (experimental)":
            settings.update(
                structure_template_dir=str(self.root / "templates" / "structures-v2"),
                structure_confidence=0.80,
                planning_preview_only=True,
            )
        if self.wall_choice.get() == "Wall network (experimental)":
            if settings.get("backend") != "hybrid":
                raise ValueError("Select a collector checkpoint before using the wall network.")
            settings.update(
                wall_model_path=str(self.root / "models" / "walls-sparse-02.pt"),
                building_model_path=str(self.root / "models" / "building-occupancy-01.pt"),
                planning_preview_only=True,
                boundary_mode="visible",
                terrain_guard=True,
                max_approach_pixels=240,
                approach_slack_pixels=32,
                ui_exclusions=[
                    (0, 0, 300, 260),
                    (1560, 0, 360, 210),
                    (0, 810, 480, 270),
                    (1640, 750, 280, 330),
                    (860, 0, 200, 110),
                ],
            )
            if (self.root / "models" / "boundary-pixels-01.pt").exists():
                settings.update(
                    boundary_edge_mode="ensemble",
                    boundary_threshold=0.2,
                    boundary_model_path=str(self.root / "models" / "boundary-pixels-01.pt"),
                )
        return settings

    def open_image(self):
        path = filedialog.askopenfilename(
            parent=self, filetypes=[("Screenshots", "*.png *.jpg *.jpeg")]
        )
        if path:
            self.load(Path(path))

    def load(self, path):
        try:
            with Image.open(path) as image:
                self.image = image.convert("RGB")
            self.record = self.dataset.frame(self.image)
            self.group.set(self.record["group"])
            scope = self.record.get(
                "reviewed_classes", KINDS if self.record.get("reviewed") else COLLECTORS
            )
            self.review_scope.set("All objects" if set(KINDS) <= set(scope) else "Collectors only")
            self.selection = self.anchor = None
            self.show_labels()
            self.status.set(
                f"{path.name}: {self.image.width} x {self.image.height}. Drag a box around one object."
            )
        except OSError as error:
            self.status.set("Capture unavailable: " + str(error))

    def show(self, image):
        preview = image.copy()
        preview.thumbnail((1100, 620))
        self.scale_x, self.scale_y = image.width / preview.width, image.height / preview.height
        self.display_size = preview.size
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def point(self, event):
        return (
            max(0, min(event.x, self.display_size[0])),
            max(0, min(event.y, self.display_size[1])),
        )

    def begin(self, event):
        if self.image is not None:
            self.show_labels()
            self.selection = None
            self.anchor = self.point(event)
            self.rectangle = self.canvas.create_rectangle(
                *self.anchor, *self.anchor, outline="lime", width=2
            )

    def drag(self, event):
        if self.anchor is not None:
            self.canvas.coords(self.rectangle, *self.anchor, *self.point(event))

    def end(self, event):
        if self.anchor is not None:
            ex, ey = self.point(event)
            sx, sy = self.anchor
            self.selection = (
                round(min(sx, ex) * self.scale_x),
                round(min(sy, ey) * self.scale_y),
                round(max(sx, ex) * self.scale_x),
                round(max(sy, ey) * self.scale_y),
            )
            self.anchor = None
            self.status.set(f"Selected {self.selection}. Choose the object type and save.")

    def save_sample(self):
        if not self.selection or self.image is None:
            return
        x1, y1, x2, y2 = self.selection
        if x2 - x1 < 6 or y2 - y1 < 6:
            messagebox.showerror(
                "Sample too small",
                "Select the complete object, at least 6 x 6 pixels.",
                parent=self,
            )
            return
        config = json.loads((self.root / "config.json").read_text())
        directory = self.root / config["vision"]["template_dir"] / self.kind.get()
        directory.mkdir(parents=True, exist_ok=True)
        name = uuid.uuid4().hex[:12] + ".png"
        self.image.crop(self.selection).save(directory / name)
        item = {"kind": self.kind.get(), "box": list(self.selection)}
        if item not in self.record["objects"]:
            self.record["objects"].append(item)
        self.save_labels()
        self.show_labels()
        self.status.set(
            f"Saved sample and training label. {len(self.record['objects'])} boxes on this frame. Label every visible supported object, then mark fully labeled."
        )

    def save_labels(self):
        self.record["group"] = self.group.get().strip()
        self.record["reviewed"] = False
        self.dataset.save(self.record)

    def show_labels(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        for item in self.record["objects"]:
            draw.rectangle(item["box"], outline="cyan", width=2)
            draw.text(
                item["box"][:2], item["kind"], fill="cyan", stroke_width=1, stroke_fill="black"
            )
        self.show(image)

    def undo_label(self):
        if self.record and self.record["objects"]:
            self.record["objects"].pop()
            self.save_labels()
            self.show_labels()
            self.status.set(
                f"{len(self.record['objects'])} labels remaining. Frame requires review again."
            )

    def remove_label(self, event):
        if self.record is None:
            return
        px, py = self.point(event)
        px, py = px * self.scale_x, py * self.scale_y
        hits = [
            item
            for item in self.record["objects"]
            if item["box"][0] <= px <= item["box"][2] and item["box"][1] <= py <= item["box"][3]
        ]
        if hits:
            item = min(
                hits,
                key=lambda item: (
                    (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1])
                ),
            )
            self.record["objects"].remove(item)
            self.save_labels()
            self.show_labels()
            self.status.set(f"Removed {item['kind']}. Frame requires review again.")

    def review(self):
        if self.record is None:
            return
        if not self.group.get().strip():
            messagebox.showerror(
                "Base ID needed",
                "Give this base an ID. Reuse it for other screenshots of the same base.",
                parent=self,
            )
            return
        self.record["group"] = self.group.get().strip()
        self.record["reviewed"] = True
        if "split" in self.record:
            self.record["grouping_reviewed"] = True
        self.record["reviewed_classes"] = list(
            COLLECTORS if self.review_scope.get() == "Collectors only" else KINDS
        )
        self.dataset.save(self.record)
        self.status.set(
            f"Frame reviewed ({self.review_scope.get()}): {len(self.record['objects'])} labels, base {self.record['group']}. Export checks the reviewed class scope."
        )

    def suggest_labels(self):
        if self.image is None:
            return
        if self.record["objects"]:
            self.status.set(
                "Suggestions only populate an unlabeled frame. Existing labels were kept."
            )
            return
        try:
            found = VisualPlanner(self.analysis_settings()).detector.detect(self.image)
            self.record["objects"] = [
                {
                    "kind": d.kind,
                    "box": [d.box[0], d.box[1], d.box[0] + d.box[2], d.box[1] + d.box[3]],
                }
                for d in found
            ]
            self.save_labels()
            self.show_labels()
            self.status.set(
                f"{len(found)} draft boxes. Remove incorrect labels with Undo and add all missing objects before marking fully labeled."
            )
        except Exception as error:
            messagebox.showerror("Suggestions", str(error), parent=self)

    def analyze(self):
        if self.image is None:
            return
        self.status.set("Analyzing...")
        self.update_idletasks()
        try:
            config = json.loads((self.root / "config.json").read_text())
            if tuple(config["capture_size"]) != self.image.size:
                raise ValueError(
                    "Use a full window capture matching capture_size, not a cropped screenshot."
                )
            plan = VisualPlanner(self.analysis_settings()).analyze_offline(self.image)
            overlay = draw_plan(self.image, plan)
            overlay.save(self.root / "vision-overlay.png")
            (self.root / "vision-plan.json").write_text(json.dumps(plan.summary(), indent=2))
            self.show(overlay)
            summary = plan.summary()
            self.status.set(
                f"OFFLINE PROPOSAL: {summary['collectors']} collectors, {summary['eligible']} estimated reachable. "
                + " ".join(summary["problems"])
                + " Green: reachable estimate; red: skip; orange: wall estimates; blue: building boxes; cyan: boundary."
            )
        except Exception as error:
            messagebox.showerror("Analysis", str(error), parent=self)
