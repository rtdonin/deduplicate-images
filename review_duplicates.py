import os
import csv
import json
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV     = os.path.join(SCRIPT_DIR, "similar_photos.csv")
DEFAULT_REMOVED = os.path.join(os.path.expanduser("~"), "Desktop", "duplicates_removed")
PROGRESS_FILE   = os.path.join(SCRIPT_DIR, "review_progress.json")


class ReviewApp:
    def __init__(self, root, pairs, removed_folder, start_index=0):
        self.root           = root
        self.pairs          = pairs
        self.removed_folder = removed_folder
        self.index          = start_index
        self.history        = []  # (index, moved_from, moved_to) - for undo on back

        os.makedirs(removed_folder, exist_ok=True)
        self._setup_ui()
        self._show_pair()

    # ── UI setup ──────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.root.title("Duplicate Photo Reviewer")
        self.root.configure(bg="#1e1e1e")

        self.progress = tk.Label(self.root, text="",
                                 font=("Arial", 12, "bold"), bg="#1e1e1e", fg="white")
        self.progress.pack(pady=(10, 4))

        img_frame = tk.Frame(self.root, bg="#1e1e1e")
        img_frame.pack(fill="both", expand=True, padx=10)

        self.img1_label, self.path1_label = self._make_panel(img_frame, "[1]  Keep this", "left")
        self.img2_label, self.path2_label = self._make_panel(img_frame, "[2]  Keep this", "right")

        tk.Label(self.root,
                 text="1 = Keep LEFT (remove right)     "
                      "2 = Keep RIGHT (remove left)     "
                      "3 = Skip - not duplicates     "
                      "B = Back     "
                      "Esc = Quit",
                 font=("Arial", 10), bg="#1e1e1e", fg="#4fc3f7"
                 ).pack(pady=8)

        self.root.bind("1",        lambda e: self._handle(1))
        self.root.bind("2",        lambda e: self._handle(2))
        self.root.bind("3",        lambda e: self._handle(3))
        self.root.bind("b",        lambda e: self._go_back())
        self.root.bind("B",        lambda e: self._go_back())
        self.root.bind("<Escape>", lambda e: self._quit())

    def _make_panel(self, parent, title, side):
        frame = tk.Frame(parent, bg="#1e1e1e")
        frame.pack(side=side, fill="both", expand=True, padx=5)

        tk.Label(frame, text=title, font=("Arial", 11, "bold"),
                 bg="#1e1e1e", fg="#66bb6a").pack(pady=(5, 2))

        img_label = tk.Label(frame, bg="#2d2d2d")
        img_label.pack()

        path_label = tk.Label(frame, text="", wraplength=560,
                              font=("Arial", 8), bg="#1e1e1e", fg="#aaaaaa")
        path_label.pack(pady=3)

        return img_label, path_label

    # ── Pair display ──────────────────────────────────────────────────────────
    def _show_pair(self):
        # Skip pairs where an image was already removed by a previous decision
        while self.index < len(self.pairs):
            p1, p2 = self.pairs[self.index]
            if os.path.exists(p1) and os.path.exists(p2):
                break
            self.index += 1

        if self.index >= len(self.pairs):
            self._clear_progress()
            messagebox.showinfo("Done", "All pairs reviewed!")
            self.root.quit()
            return

        p1, p2 = self.pairs[self.index]
        self.progress.config(text=f"Pair {self.index + 1} of {len(self.pairs)}")

        self._load_image(p1, self.img1_label, self.path1_label)
        self._load_image(p2, self.img2_label, self.path2_label)

    def _load_image(self, path, img_label, path_label):
        try:
            img = Image.open(path)
            img.thumbnail((580, 520), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            img_label.config(image=photo, text="")
            img_label.image = photo          # keep reference - prevents garbage collection
        except Exception:
            img_label.config(image="", text="[Cannot load image]",
                             font=("Arial", 10), fg="red")
            img_label.image = None
        path_label.config(text=path)

    # ── Key handling ──────────────────────────────────────────────────────────
    def _handle(self, choice):
        if self.index >= len(self.pairs):
            return

        p1, p2       = self.pairs[self.index]
        moved_from   = None
        moved_to     = None

        if choice == 1:
            moved_from, moved_to = self._move(p2)   # keep left, remove right
        elif choice == 2:
            moved_from, moved_to = self._move(p1)   # keep right, remove left
        # choice == 3: skip, keep both

        self.history.append((self.index, moved_from, moved_to))
        self.index += 1
        self._save_progress()
        self._show_pair()

    def _go_back(self):
        if not self.history:
            return

        prev_index, moved_from, moved_to = self.history.pop()

        # Undo the file move if one was made this session
        if moved_from and moved_to and os.path.exists(moved_to):
            try:
                shutil.move(moved_to, moved_from)
            except Exception as e:
                messagebox.showerror("Error", f"Could not restore file:\n{moved_from}\n\n{e}")
                return

        self.index = prev_index
        self._save_progress()
        self._show_pair()

    def _move(self, filepath):
        try:
            dest = os.path.join(self.removed_folder, Path(filepath).name)
            if os.path.exists(dest):
                stem   = Path(filepath).stem
                suffix = Path(filepath).suffix
                dest   = os.path.join(self.removed_folder,
                                      f"{stem}_{abs(hash(filepath)) % 99999}{suffix}")
            shutil.move(filepath, dest)
            return filepath, dest
        except Exception as e:
            messagebox.showerror("Error", f"Could not move file:\n{filepath}\n\n{e}")
            return None, None

    # ── Progress save / load ──────────────────────────────────────────────────
    def _save_progress(self):
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"index": self.index}, f)

    def _clear_progress(self):
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

    def _quit(self):
        self._save_progress()
        self.root.quit()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactively review duplicate photo pairs from find_similar_photos.py."
    )
    parser.add_argument("csv_file", nargs="?", default=DEFAULT_CSV,
                        help="CSV file to review (default: same folder as script)")
    parser.add_argument("--removed-folder", default=DEFAULT_REMOVED,
                        help="Folder to move removed files into (default: Desktop/duplicates_removed)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"CSV file not found: {args.csv_file}")
        sys.exit(1)

    pairs = []
    with open(args.csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[0], row[1]))

    if not pairs:
        print("No pairs found in CSV.")
        sys.exit(0)

    # Check for saved progress and offer to resume
    start_index = 0
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                saved = json.load(f).get("index", 0)
            if saved > 0:
                root_temp = tk.Tk()
                root_temp.withdraw()
                resume = messagebox.askyesno(
                    "Resume Session",
                    f"Previous session found.\nYou were at pair {saved + 1} of {len(pairs)}.\n\nResume from there?"
                )
                root_temp.destroy()
                if resume:
                    start_index = saved
        except Exception:
            pass

    print(f"Loaded {len(pairs)} pairs. Starting at pair {start_index + 1}.")
    print(f"Removed files will be moved to: {args.removed_folder}")
    print("They are NOT permanently deleted - you can restore them from that folder.")

    root = tk.Tk()
    root.geometry("1300x750")
    ReviewApp(root, pairs, args.removed_folder, start_index)
    root.mainloop()


if __name__ == "__main__":
    main()
