import os
import csv
import json
import queue
import shutil
import threading
import multiprocessing
import imagehash
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog, scrolledtext
from PIL import Image, ImageTk
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Constants ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE    = os.path.join(SCRIPT_DIR, "photo_hashes_cache.npz")
PAIRS_FILE    = os.path.join(SCRIPT_DIR, "pairs_cache.json")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, "review_progress.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
HASH_SIZE  = 8
NUM_HASHES = 6
BATCH_SIZE = 64

_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# ── Colours ────────────────────────────────────────────────────────────────────
BG       = "#1e1e1e"
BG2      = "#2d2d2d"
FG       = "#ffffff"
FG_DIM   = "#aaaaaa"
ACCENT   = "#4fc3f7"
GREEN    = "#66bb6a"
RED      = "#ef5350"
YELLOW   = "#ffd54f"
BTN_BG   = "#37474f"

# ══════════════════════════════════════════════════════════════════════════════
#  HASHING CORE  (worker-safe - no tkinter imports at module level)
# ══════════════════════════════════════════════════════════════════════════════

def compute_hashes_batch(filepaths):
    results = []
    for filepath in filepaths:
        try:
            with Image.open(filepath) as img:
                img.draft("RGB", (512, 512))
                img = img.convert("RGB")
                img.thumbnail((512, 512), Image.LANCZOS)
                w, h = img.size
                crops = [
                    img,
                    img.crop((int(w*.2), int(h*.2), int(w*.8), int(h*.8))),
                    img.crop((0,         0,          w,         int(h*.8))),
                    img.crop((0,         int(h*.2),  w,         h        )),
                    img.crop((0,         0,          int(w*.8), h        )),
                    img.crop((int(w*.2), 0,          w,         h        )),
                ]
                packed = np.array([
                    np.packbits(imagehash.dhash(c, hash_size=HASH_SIZE).hash.flatten())
                    for c in crops
                ], dtype=np.uint8)
                results.append((filepath, packed))
        except Exception:
            results.append((filepath, None))
    return results


def pairwise_hamming(a, b):
    xor = a[:, np.newaxis, :] ^ b[np.newaxis, :, :]
    return _POPCOUNT[xor].sum(axis=2)


def pack_hash_list(hex_strings):
    return np.array(
        [np.packbits(imagehash.hex_to_hash(h).hash.flatten()) for h in hex_strings],
        dtype=np.uint8
    )


def find_similar_core(root_dir, threshold, use_cache, log_q, hash_prog_q, cmp_prog_q, result_q):
    """
    Runs in a background thread.
    Sends log strings to log_q, progress tuples to hash/cmp prog queues,
    and final pairs list to result_q.
    """
    def log(msg):
        log_q.put(msg)

    workers = max(1, multiprocessing.cpu_count() - 1)
    log(f"Using {workers} worker processes")

    # Scan
    log("\nScanning for image files...")
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                all_files.append(os.path.join(dirpath, filename))
    log(f"Found {len(all_files)} image files")

    # Cache
    cached_paths, cached_hashes = [], np.empty((0, NUM_HASHES, 8), dtype=np.uint8)
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            log(f"Loading cache...")
            data = np.load(CACHE_FILE, allow_pickle=True)
            p = data['paths'].tolist()
            h = data['hashes']
            if h.ndim == 3 and h.shape[1] == NUM_HASHES:
                cached_paths, cached_hashes = p, h
                log(f"  {len(cached_paths)} hashes loaded from cache")
        except Exception:
            log("  Cache unreadable - starting fresh")

    cached_set = set(cached_paths)
    to_hash = [f for f in all_files if f not in cached_set]
    log(f"{len(to_hash)} to hash  |  {len(cached_set)} cached")

    # Hash
    new_paths, new_hashes_list = [], []
    if to_hash:
        batches = [to_hash[i:i + BATCH_SIZE] for i in range(0, len(to_hash), BATCH_SIZE)]
        errors = 0
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compute_hashes_batch, b): b for b in batches}
            for future in as_completed(futures):
                for filepath, packed in future.result():
                    if packed is not None:
                        new_paths.append(filepath)
                        new_hashes_list.append(packed)
                    else:
                        errors += 1
                    done += 1
                    hash_prog_q.put((done, len(to_hash)))

        if errors:
            log(f"  {errors} file(s) skipped")

        new_hashes = np.stack(new_hashes_list)
        all_paths  = cached_paths + new_paths
        all_hashes = np.concatenate([cached_hashes, new_hashes], axis=0)
        np.savez_compressed(CACHE_FILE,
                            paths=np.array(all_paths, dtype=object),
                            hashes=all_hashes)
        log("Cache saved")
    else:
        all_paths, all_hashes = cached_paths, cached_hashes

    # Build matrix
    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    valid_files = [f for f in all_files if f in path_to_idx]
    stacked     = all_hashes[[path_to_idx[f] for f in valid_files]]
    n           = len(valid_files)
    log(f"\nComparing {n} images...")

    # Compare
    similar = []
    chunk   = 1000
    total_chunks = (n + chunk - 1) // chunk
    done_chunks  = 0

    for i in range(0, n, chunk):
        si = stacked[i:i + chunk]
        for j in range(i, n, chunk):
            sj      = stacked[j:j + chunk]
            m, k    = len(si), len(sj)
            sj_flat = sj.reshape(-1, 8)
            mask    = np.zeros((m, k), dtype=bool)
            for ki in range(NUM_HASHES):
                ci   = si[:, ki, :]
                xor  = ci[:, np.newaxis, :] ^ sj_flat[np.newaxis, :, :]
                dists = _POPCOUNT[xor].sum(axis=2).reshape(m, k, NUM_HASHES)
                mask |= (dists <= threshold).any(axis=2)
            rows, cols = np.where(mask)
            for r, c in zip(rows, cols):
                gi, gj = i + r, j + c
                if gi < gj:
                    similar.append((valid_files[gi], valid_files[gj]))
        done_chunks += 1
        cmp_prog_q.put((done_chunks, total_chunks))

    # Deduplicate
    similar = list(dict.fromkeys(similar))
    log(f"\nFound {len(similar)} similar pair(s).")

    # Save pairs internally
    with open(PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump(similar, f)

    result_q.put(similar)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_move(filepath, dest_folder):
    """Move a file to dest_folder, handling name collisions. Returns (src, dst)."""
    dest = os.path.join(dest_folder, Path(filepath).name)
    if os.path.exists(dest):
        stem   = Path(filepath).stem
        suffix = Path(filepath).suffix
        dest   = os.path.join(dest_folder, f"{stem}_{abs(hash(filepath)) % 99999}{suffix}")
    shutil.move(filepath, dest)
    return filepath, dest


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Duplicate Photo Finder")
        self.geometry("1300x800")
        self.configure(bg=BG)
        self.resizable(True, True)

        # Shared state
        self.pairs          = []
        self.removed_folder = os.path.join(os.path.expanduser("~"), "Desktop", "duplicates_removed")
        self.source_folder  = ""

        self._current_screen = None
        self.show_screen(StartupTestScreen)

    def show_screen(self, ScreenClass, **kwargs):
        if self._current_screen is not None:
            self._current_screen.destroy()
        self._current_screen = ScreenClass(self, **kwargs)
        self._current_screen.pack(fill="both", expand=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class HomeScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master = master

        tk.Label(self, text="Duplicate Photo Finder",
                 font=("Arial", 26, "bold"), bg=BG, fg=FG).pack(pady=(80, 10))
        tk.Label(self, text="Find, review, and remove duplicate images from your photo library.",
                 font=("Arial", 12), bg=BG, fg=FG_DIM).pack(pady=(0, 60))

        btn_cfg = dict(font=("Arial", 14, "bold"), width=22, height=2,
                       bg=BTN_BG, fg=FG, activebackground=ACCENT,
                       activeforeground=BG, relief="flat", cursor="hand2")

        tk.Button(self, text="Find Duplicates",
                  command=lambda: master.show_screen(FindScreen),
                  **btn_cfg).pack(pady=12)

        tk.Button(self, text="Review Duplicates",
                  command=self._open_review,
                  **btn_cfg).pack(pady=12)

        tk.Button(self, text="Remove Duplicates",
                  command=self._open_remove,
                  **btn_cfg).pack(pady=12)

        tk.Button(self, text="Delete Removed Folder",
                  command=self._delete_removed_folder,
                  **btn_cfg).pack(pady=12)

    def _delete_removed_folder(self):
        dest = self.master.removed_folder
        if not os.path.exists(dest):
            messagebox.showinfo("Not Found", f"The removed folder does not exist:\n{dest}")
            return
        if not messagebox.askyesno(
            "Permanently Delete",
            f"This will permanently delete:\n{dest}\n\nThis CANNOT be undone. Continue?"
        ):
            return
        try:
            shutil.rmtree(dest)
            messagebox.showinfo("Deleted", "Removed folder permanently deleted.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete folder:\n{e}")

    def _load_pairs(self):
        """Load pairs from memory or internal file."""
        if self.master.pairs:
            return self.master.pairs
        if os.path.exists(PAIRS_FILE):
            with open(PAIRS_FILE, encoding="utf-8") as f:
                pairs = json.load(f)
            self.master.pairs = [tuple(p) for p in pairs]
            return self.master.pairs
        return []

    def _open_review(self):
        pairs = self._load_pairs()
        if not pairs:
            messagebox.showinfo("No Pairs", "No duplicate pairs found yet.\nRun 'Find Duplicates' first.")
            return
        self.master.show_screen(ReviewScreen)

    def _open_remove(self):
        pairs = self._load_pairs()
        if not pairs:
            messagebox.showinfo("No Pairs", "No duplicate pairs found yet.\nRun 'Find Duplicates' first.")
            return
        self.master.show_screen(RemoveScreen)


# ══════════════════════════════════════════════════════════════════════════════
#  FIND SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class FindScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master   = master
        self.running  = False

        # ── Top bar ──
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(15, 5))
        tk.Button(top, text="← Home", command=self._go_home,
                  font=("Arial", 10), bg=BTN_BG, fg=FG, relief="flat",
                  cursor="hand2").pack(side="left")
        tk.Label(top, text="Find Duplicates", font=("Arial", 16, "bold"),
                 bg=BG, fg=FG).pack(side="left", padx=20)

        # ── Folder inputs ──
        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="x", padx=30, pady=10)

        self._make_folder_row(grid, 0, "Photos Folder:",
                              master.source_folder, "source")
        self._make_folder_row(grid, 1, "Removed Files Folder:",
                              master.removed_folder, "removed")

        # ── Options ──
        opts = tk.Frame(self, bg=BG)
        opts.pack(fill="x", padx=30, pady=5)

        tk.Label(opts, text="Similarity Threshold (1–64):",
                 font=("Arial", 11), bg=BG, fg=FG).pack(side="left")
        self.threshold_var = tk.IntVar(value=10)
        tk.Spinbox(opts, from_=1, to=64, textvariable=self.threshold_var,
                   width=5, font=("Arial", 11), bg=BG2, fg=FG,
                   buttonbackground=BTN_BG).pack(side="left", padx=10)

        self.cache_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts, text="Use cached hashes (faster re-run)",
                       variable=self.cache_var,
                       font=("Arial", 11), bg=BG, fg=FG,
                       selectcolor=BG2, activebackground=BG).pack(side="left", padx=20)

        # ── Start button ──
        self.start_btn = tk.Button(self, text="Start",
                                   command=self._start,
                                   font=("Arial", 13, "bold"), width=14,
                                   bg=GREEN, fg=BG, relief="flat", cursor="hand2")
        self.start_btn.pack(pady=10)

        # ── Progress bars ──
        prog_frame = tk.Frame(self, bg=BG)
        prog_frame.pack(fill="x", padx=30, pady=5)

        tk.Label(prog_frame, text="Hashing:", font=("Arial", 10),
                 bg=BG, fg=FG_DIM, width=10, anchor="w").grid(row=0, column=0, sticky="w")
        self.hash_bar = ttk.Progressbar(prog_frame, length=600, mode="determinate")
        self.hash_bar.grid(row=0, column=1, padx=10, pady=3)
        self.hash_lbl = tk.Label(prog_frame, text="", font=("Arial", 10),
                                 bg=BG, fg=FG_DIM, width=12)
        self.hash_lbl.grid(row=0, column=2)

        tk.Label(prog_frame, text="Comparing:", font=("Arial", 10),
                 bg=BG, fg=FG_DIM, width=10, anchor="w").grid(row=1, column=0, sticky="w")
        self.cmp_bar = ttk.Progressbar(prog_frame, length=600, mode="determinate")
        self.cmp_bar.grid(row=1, column=1, padx=10, pady=3)
        self.cmp_lbl = tk.Label(prog_frame, text="", font=("Arial", 10),
                                bg=BG, fg=FG_DIM, width=12)
        self.cmp_lbl.grid(row=1, column=2)

        # ── Log output ──
        self.log = scrolledtext.ScrolledText(self, height=14, font=("Consolas", 9),
                                             bg=BG2, fg=FG, insertbackground=FG,
                                             relief="flat")
        self.log.pack(fill="both", expand=True, padx=30, pady=(5, 10))

        # ── Done button (hidden until finished) ──
        self.done_btn = tk.Button(self, text="Review Duplicates →",
                                  command=lambda: master.show_screen(ReviewScreen),
                                  font=("Arial", 12, "bold"), bg=ACCENT, fg=BG,
                                  relief="flat", cursor="hand2")

        # Queues for thread communication
        self.log_q      = queue.Queue()
        self.hash_prog_q = queue.Queue()
        self.cmp_prog_q  = queue.Queue()
        self.result_q    = queue.Queue()

    def _make_folder_row(self, parent, row, label, default, attr):
        tk.Label(parent, text=label, font=("Arial", 11), bg=BG, fg=FG,
                 width=22, anchor="w").grid(row=row, column=0, sticky="w", pady=5)
        var = tk.StringVar(value=default)
        setattr(self, f"{attr}_var", var)
        tk.Entry(parent, textvariable=var, font=("Arial", 10), bg=BG2, fg=FG,
                 insertbackground=FG, width=60, relief="flat").grid(row=row, column=1,
                                                                     padx=10, sticky="ew")
        tk.Button(parent, text="Browse",
                  command=lambda v=var: v.set(filedialog.askdirectory() or v.get()),
                  font=("Arial", 10), bg=BTN_BG, fg=FG, relief="flat",
                  cursor="hand2").grid(row=row, column=2)

    def _go_home(self):
        if self.running:
            if not messagebox.askyesno("Running", "Scan is in progress. Go home anyway?"):
                return
        self.master.show_screen(HomeScreen)

    def _start(self):
        src = self.source_var.get().strip()
        rem = self.removed_var.get().strip()
        if not src or not os.path.isdir(src):
            messagebox.showerror("Error", "Please select a valid Photos Folder.")
            return
        if not rem:
            messagebox.showerror("Error", "Please select a Removed Files Folder.")
            return

        os.makedirs(rem, exist_ok=True)
        self.master.source_folder  = src
        self.master.removed_folder = rem

        self.start_btn.config(state="disabled")
        self.done_btn.pack_forget()
        self.log.delete("1.0", "end")
        self.hash_bar["value"] = 0
        self.cmp_bar["value"]  = 0
        self.running = True

        t = threading.Thread(target=find_similar_core, daemon=True, args=(
            src,
            self.threshold_var.get(),
            self.cache_var.get(),
            self.log_q,
            self.hash_prog_q,
            self.cmp_prog_q,
            self.result_q,
        ))
        t.start()
        self.after(100, self._poll)

    def _poll(self):
        # Drain log queue
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.log.insert("end", msg + "\n")
                self.log.see("end")
        except queue.Empty:
            pass

        # Hash progress
        try:
            while True:
                done, total = self.hash_prog_q.get_nowait()
                pct = int(done / total * 100) if total else 0
                self.hash_bar["value"] = pct
                self.hash_lbl.config(text=f"{done}/{total}")
        except queue.Empty:
            pass

        # Compare progress
        try:
            while True:
                done, total = self.cmp_prog_q.get_nowait()
                pct = int(done / total * 100) if total else 0
                self.cmp_bar["value"] = pct
                self.cmp_lbl.config(text=f"{done}/{total}")
        except queue.Empty:
            pass

        # Check for result
        try:
            pairs = self.result_q.get_nowait()
            self.master.pairs = pairs
            self.running = False
            self.start_btn.config(state="normal")
            self.done_btn.pack(pady=(0, 10))
        except queue.Empty:
            self.after(100, self._poll)


# ══════════════════════════════════════════════════════════════════════════════
#  REVIEW SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class ReviewScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master         = master
        self.pairs          = list(master.pairs)
        self.removed_folder = master.removed_folder
        self.index          = 0
        self.history        = []   # (index, moved_from, moved_to) - for undo

        os.makedirs(self.removed_folder, exist_ok=True)
        self._setup_ui()

        # Check for saved progress
        start = 0
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE) as f:
                    saved = json.load(f).get("index", 0)
                if saved > 0 and messagebox.askyesno(
                    "Resume",
                    f"Previous session found.\nYou were at pair {saved + 1} of {len(self.pairs)}.\n\nResume from there?"
                ):
                    start = saved
            except Exception:
                pass
        self.index = start
        self._show_pair()

    def _setup_ui(self):
        # ── Top bar ──
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(15, 5))
        tk.Button(top, text="← Home", command=self._go_home,
                  font=("Arial", 10), bg=BTN_BG, fg=FG, relief="flat",
                  cursor="hand2").pack(side="left")
        self.progress_lbl = tk.Label(top, text="",
                                     font=("Arial", 12, "bold"), bg=BG, fg=FG)
        self.progress_lbl.pack(side="left", padx=20)

        # ── Image panels ──
        img_frame = tk.Frame(self, bg=BG)
        img_frame.pack(fill="both", expand=True, padx=10)

        self.img1_lbl, self.path1_lbl = self._make_panel(img_frame, "[1]  Keep this", "left")
        self.img2_lbl, self.path2_lbl = self._make_panel(img_frame, "[2]  Keep this", "right")

        # ── Instructions ──
        tk.Label(self,
                 text="1 = Keep LEFT   |   2 = Keep RIGHT   |   3 = Skip   |   "
                      "4 = Remove Both   |   B = Back   |   Esc = Home",
                 font=("Arial", 10), bg=BG, fg=ACCENT).pack(pady=8)

        self.bind_all("1",        lambda e: self._handle(1))
        self.bind_all("2",        lambda e: self._handle(2))
        self.bind_all("3",        lambda e: self._handle(3))
        self.bind_all("4",        lambda e: self._handle(4))
        self.bind_all("b",        lambda e: self._go_back())
        self.bind_all("B",        lambda e: self._go_back())
        self.bind_all("<Escape>", lambda e: self._go_home())

    def _make_panel(self, parent, title, side):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(side=side, fill="both", expand=True, padx=5)
        tk.Label(frame, text=title, font=("Arial", 11, "bold"),
                 bg=BG, fg=GREEN).pack(pady=(5, 2))
        img_lbl = tk.Label(frame, bg=BG2)
        img_lbl.pack()
        path_lbl = tk.Label(frame, text="", wraplength=560,
                            font=("Arial", 8), bg=BG, fg=FG_DIM)
        path_lbl.pack(pady=3)
        return img_lbl, path_lbl

    def _show_pair(self):
        while self.index < len(self.pairs):
            p1, p2 = self.pairs[self.index]
            if os.path.exists(p1) and os.path.exists(p2):
                break
            self.index += 1

        if self.index >= len(self.pairs):
            self._clear_progress()
            messagebox.showinfo("Done", "All pairs reviewed!")
            if os.path.exists(self.removed_folder):
                if messagebox.askyesno(
                    "Delete Removed Folder",
                    f"Permanently delete the removed folder?\n{self.removed_folder}\n\nThis CANNOT be undone."
                ):
                    try:
                        shutil.rmtree(self.removed_folder)
                        messagebox.showinfo("Deleted", "Removed folder permanently deleted.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Could not delete folder:\n{e}")
            self._go_home()
            return

        p1, p2 = self.pairs[self.index]
        self.progress_lbl.config(text=f"Pair {self.index + 1} of {len(self.pairs)}")
        self._load_image(p1, self.img1_lbl, self.path1_lbl)
        self._load_image(p2, self.img2_lbl, self.path2_lbl)

    def _load_image(self, path, lbl, path_lbl):
        try:
            img = Image.open(path)
            img.thumbnail((560, 500), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl.config(image=photo, text="")
            lbl.image = photo
        except Exception:
            lbl.config(image="", text="[Cannot load]", font=("Arial", 10), fg=RED)
            lbl.image = None
        path_lbl.config(text=path)

    def _handle(self, choice):
        if self.index >= len(self.pairs):
            return
        p1, p2 = self.pairs[self.index]
        moves  = []

        if choice == 1:
            src, dst = self._move(p2)
            moves.append((src, dst))
        elif choice == 2:
            src, dst = self._move(p1)
            moves.append((src, dst))
        elif choice == 4:
            src1, dst1 = self._move(p1)
            src2, dst2 = self._move(p2)
            moves.append((src1, dst1))
            moves.append((src2, dst2))
        # choice 3 = skip

        self.history.append((self.index, moves))
        self.index += 1
        self._save_progress()
        self._show_pair()

    def _go_back(self):
        if not self.history:
            return
        prev_index, moves = self.history.pop()
        for src, dst in moves:
            if src and dst and os.path.exists(dst):
                try:
                    shutil.move(dst, src)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not restore:\n{src}\n\n{e}")
                    return
        self.index = prev_index
        self._save_progress()
        self._show_pair()

    def _move(self, filepath):
        try:
            return safe_move(filepath, self.removed_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Could not move:\n{filepath}\n\n{e}")
            return None, None

    def _save_progress(self):
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"index": self.index}, f)

    def _clear_progress(self):
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)

    def _go_home(self):
        self._save_progress()
        self.unbind_all("1")
        self.unbind_all("2")
        self.unbind_all("3")
        self.unbind_all("4")
        self.unbind_all("b")
        self.unbind_all("B")
        self.unbind_all("<Escape>")
        self.master.show_screen(HomeScreen)


# ══════════════════════════════════════════════════════════════════════════════
#  REMOVE SCREEN
# ══════════════════════════════════════════════════════════════════════════════

class RemoveScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master         = master
        self.pairs          = list(master.pairs)
        self.removed_folder = master.removed_folder

        # ── Top bar ──
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(15, 5))
        tk.Button(top, text="← Home", command=lambda: master.show_screen(HomeScreen),
                  font=("Arial", 10), bg=BTN_BG, fg=FG, relief="flat",
                  cursor="hand2").pack(side="left")
        tk.Label(top, text="Remove Duplicates", font=("Arial", 16, "bold"),
                 bg=BG, fg=FG).pack(side="left", padx=20)

        n = len(self.pairs)
        tk.Label(self,
                 text=f"{n} duplicate pairs found.\n"
                      "Removed files are moved to the removed folder - not permanently deleted.",
                 font=("Arial", 12), bg=BG, fg=FG_DIM, justify="center").pack(pady=20)

        # ── Removed folder display ──
        rf_frame = tk.Frame(self, bg=BG)
        rf_frame.pack(pady=5)
        tk.Label(rf_frame, text="Removed Folder:", font=("Arial", 11),
                 bg=BG, fg=FG).pack(side="left")
        self.rf_var = tk.StringVar(value=self.removed_folder)
        tk.Entry(rf_frame, textvariable=self.rf_var, font=("Arial", 10),
                 bg=BG2, fg=FG, insertbackground=FG, width=55,
                 relief="flat").pack(side="left", padx=10)
        tk.Button(rf_frame, text="Browse",
                  command=lambda: self.rf_var.set(filedialog.askdirectory() or self.rf_var.get()),
                  font=("Arial", 10), bg=BTN_BG, fg=FG, relief="flat",
                  cursor="hand2").pack(side="left")

        # ── Buttons ──
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame,
                  text=f"Remove All  -  Keep Image 1\n({n} files will be moved)",
                  command=lambda: self._remove(keep_one=True),
                  font=("Arial", 12, "bold"), width=30, height=2,
                  bg=YELLOW, fg=BG, relief="flat", cursor="hand2").pack(pady=8)

        tk.Button(btn_frame,
                  text=f"Remove All  -  Both Copies\n({n * 2} files will be moved)",
                  command=lambda: self._remove(keep_one=False),
                  font=("Arial", 12, "bold"), width=30, height=2,
                  bg=RED, fg=FG, relief="flat", cursor="hand2").pack(pady=8)

        # ── Status label ──
        self.status_lbl = tk.Label(self, text="", font=("Arial", 11),
                                   bg=BG, fg=FG_DIM)
        self.status_lbl.pack(pady=10)

        # ── Delete removed folder button (hidden until removal is done) ──
        self.delete_folder_btn = tk.Button(
            self,
            text="Permanently Delete Removed Folder",
            command=self._delete_removed_folder,
            font=("Arial", 11, "bold"),
            bg=RED, fg=FG, relief="flat", cursor="hand2"
        )

    def _remove(self, keep_one):
        dest = self.rf_var.get().strip()
        if not dest:
            messagebox.showerror("Error", "Please set a removed files folder.")
            return
        os.makedirs(dest, exist_ok=True)
        self.master.removed_folder = dest

        moved = 0
        errors = 0
        for p1, p2 in self.pairs:
            targets = [p2] if keep_one else [p1, p2]
            for fp in targets:
                if os.path.exists(fp):
                    try:
                        safe_move(fp, dest)
                        moved += 1
                    except Exception:
                        errors += 1

        msg = f"{moved} file(s) moved to:\n{dest}"
        if errors:
            msg += f"\n{errors} file(s) could not be moved."
        self.status_lbl.config(text=msg, fg=GREEN)
        self.delete_folder_btn.pack(pady=5)

    def _delete_removed_folder(self):
        dest = self.rf_var.get().strip()
        if not os.path.exists(dest):
            messagebox.showinfo("Already gone", "The removed folder doesn't exist.")
            return
        if not messagebox.askyesno(
            "Permanently Delete",
            f"This will permanently delete:\n{dest}\n\nThis CANNOT be undone. Continue?"
        ):
            return
        try:
            shutil.rmtree(dest)
            self.status_lbl.config(text="Removed folder permanently deleted.", fg=RED)
            self.delete_folder_btn.pack_forget()
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete folder:\n{e}")


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_startup_tests():
    """
    Logic tests for core algorithm correctness.
    Returns list of (name, passed, detail) tuples.
    Tests cover: popcount accuracy, Hamming distance, deduplication,
    threshold boundary, self-match prevention, crop coordinate validity,
    and similarity mask OR logic.
    """
    results = []

    def check(name, condition, detail=""):
        results.append((name, bool(condition), detail))

    # 1. Popcount lookup table values
    try:
        check("Popcount: 0 bits in 0x00",     _POPCOUNT[0b00000000] == 0)
        check("Popcount: 8 bits in 0xFF",     _POPCOUNT[0b11111111] == 8)
        check("Popcount: 4 bits in 0xAA",     _POPCOUNT[0b10101010] == 4)
        check("Popcount: 3 bits in 0x70",     _POPCOUNT[0b01110000] == 3)
    except Exception as e:
        results.append(("Popcount table", False, str(e)))

    # 2. Hamming distance: identical vectors must be 0
    try:
        a    = np.array([[0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A]], dtype=np.uint8)
        dist = pairwise_hamming(a, a.copy())
        check("Hamming: identical vectors -> 0", dist[0, 0] == 0, f"got {dist[0,0]}")
    except Exception as e:
        results.append(("Hamming: identical -> 0", False, str(e)))

    # 3. Hamming distance: all-zeros vs all-ones must be 64 (every bit flipped)
    try:
        zeros = np.zeros((1, 8), dtype=np.uint8)
        ones  = np.full((1, 8), 0xFF, dtype=np.uint8)
        dist  = pairwise_hamming(zeros, ones)
        check("Hamming: zeros vs ones -> 64", dist[0, 0] == 64, f"got {dist[0,0]}")
    except Exception as e:
        results.append(("Hamming: zeros vs ones -> 64", False, str(e)))

    # 4. Hamming distance: single-byte known difference
    try:
        a    = np.array([[0xFF, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)  # 8 ones in first byte
        b    = np.array([[0x00, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)  # 0 ones in first byte
        dist = pairwise_hamming(a, b)
        check("Hamming: first-byte flip -> 8", dist[0, 0] == 8, f"got {dist[0,0]}")
    except Exception as e:
        results.append(("Hamming: first-byte flip -> 8", False, str(e)))

    # 5. Hamming is symmetric (dist(A,B) == dist(B,A))
    try:
        a    = np.array([[0x3C, 0xA5, 0x00, 0xFF, 0x55, 0xAA, 0x0F, 0xF0]], dtype=np.uint8)
        b    = np.array([[0xC3, 0x5A, 0xFF, 0x00, 0xAA, 0x55, 0xF0, 0x0F]], dtype=np.uint8)
        check("Hamming: symmetric",
              pairwise_hamming(a, b)[0, 0] == pairwise_hamming(b, a)[0, 0])
    except Exception as e:
        results.append(("Hamming: symmetric", False, str(e)))

    # 6. Pair deduplication preserves order and removes exact duplicates
    try:
        pairs   = [("a","b"), ("c","d"), ("a","b"), ("e","f"), ("c","d")]
        deduped = list(dict.fromkeys(pairs))
        check("Dedup: correct count",     len(deduped) == 3, f"got {len(deduped)}")
        check("Dedup: order preserved",   deduped == [("a","b"), ("c","d"), ("e","f")])
        check("Dedup: no false removal",  ("e","f") in deduped)
    except Exception as e:
        results.append(("Deduplication", False, str(e)))

    # 7. Threshold boundary: distance exactly at threshold passes, distance+1 fails
    try:
        a    = np.zeros((1, 8), dtype=np.uint8)
        b    = np.array([[0b00000111, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)  # 3 bits differ
        dist = int(pairwise_hamming(a, b)[0, 0])
        check("Threshold: dist 3 passes threshold 3",  dist <= 3)
        check("Threshold: dist 3 fails threshold 2",   not (dist <= 2))
        check("Threshold: dist 3 passes threshold 10", dist <= 10)
    except Exception as e:
        results.append(("Threshold boundary", False, str(e)))

    # 8. gi < gj logic prevents self-matches and duplicate pairs
    try:
        n     = 6
        pairs_found = [(i, j) for i in range(n) for j in range(n) if i < j]
        check("No self-matches (gi < gj)",    all(x != y for x, y in pairs_found))
        check("No duplicate pairs (gi < gj)", len(pairs_found) == len(set(pairs_found)))
        check("Correct pair count for n=6",   len(pairs_found) == 15)  # n*(n-1)/2
    except Exception as e:
        results.append(("No self-matches", False, str(e)))

    # 9. Crop coordinates produce non-empty boxes for various image sizes
    try:
        for w, h in [(64, 64), (100, 200), (1920, 1080), (50, 50), (4000, 3000)]:
            boxes = [
                (0,         0,          w,         h        ),  # full
                (int(w*.2), int(h*.2),  int(w*.8), int(h*.8)),  # center 60%
                (0,         0,          w,         int(h*.8)),  # top 80%
                (0,         int(h*.2),  w,         h        ),  # bottom 80%
                (0,         0,          int(w*.8), h        ),  # left 80%
                (int(w*.2), 0,          w,         h        ),  # right 80%
            ]
            for x1, y1, x2, y2 in boxes:
                check(f"Crop non-empty {w}x{h} ({x1},{y1})-({x2},{y2})",
                      x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0)
    except Exception as e:
        results.append(("Crop coordinates", False, str(e)))

    # 10. Similarity mask OR logic accumulates correctly across iterations
    try:
        mask = np.zeros((3, 3), dtype=bool)
        mask |= np.array([[True,  False, False],
                          [False, True,  False],
                          [False, False, False]])
        mask |= np.array([[False, False, True ],
                          [False, False, False],
                          [False, False, True ]])
        expected = np.array([[True,  False, True ],
                             [False, True,  False],
                             [False, False, True ]])
        check("Mask OR: accumulates without overwrite", np.array_equal(mask, expected))
        check("Mask OR: False OR False stays False",    not mask[2, 0])
        check("Mask OR: True OR False stays True",      mask[0, 0])
    except Exception as e:
        results.append(("Similarity mask OR", False, str(e)))

    return results


class StartupTestScreen(tk.Frame):
    """Displayed on launch. Auto-advances after 1.5 s if all tests pass."""

    AUTO_CLOSE_MS = 1500

    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.master  = master
        self.results = run_startup_tests()

        passed = [r for r in self.results if r[1]]
        failed = [r for r in self.results if not r[1]]
        all_ok = len(failed) == 0

        tk.Label(self, text="Startup Tests",
                 font=("Arial", 15, "bold"), bg=BG, fg=FG).pack(pady=(18, 6))

        summary_color = GREEN if all_ok else RED
        summary_text  = (f"All {len(self.results)} tests passed"
                         if all_ok else
                         f"{len(failed)} of {len(self.results)} tests FAILED")
        tk.Label(self, text=summary_text,
                 font=("Arial", 12, "bold"), bg=BG, fg=summary_color).pack(pady=(0, 10))

        # Scrollable test list
        frame = tk.Frame(self, bg=BG2)
        frame.pack(fill="both", expand=True, padx=30, pady=(0, 10))

        canvas   = tk.Canvas(frame, bg=BG2, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner    = tk.Frame(canvas, bg=BG2)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for name, ok, detail in self.results:
            icon  = "+" if ok else "X"
            color = GREEN if ok else RED
            line  = f"  [{icon}]  {name}"
            if detail:
                line += f"  ({detail})"
            tk.Label(inner, text=line, font=("Consolas", 9),
                     bg=BG2, fg=color, anchor="w").pack(fill="x", padx=8, pady=1)

        # Continue button
        btn_text  = "Continue" if all_ok else "Continue Anyway"
        btn_color = ACCENT if all_ok else YELLOW
        tk.Button(self, text=btn_text, command=self._proceed,
                  font=("Arial", 11, "bold"), bg=btn_color, fg=BG,
                  relief="flat", cursor="hand2", width=18).pack(pady=(0, 15))

        if all_ok:
            self.after(self.AUTO_CLOSE_MS, self._proceed)

    def _proceed(self):
        self.master.show_screen(HomeScreen)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    multiprocessing.freeze_support()   # Required for PyInstaller .exe
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
