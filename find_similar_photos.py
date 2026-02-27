import os
import csv
import multiprocessing
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
CACHE_FILE = os.path.join(SCRIPT_DIR, "photo_hashes_cache.npz")   # Binary format - faster than JSON
HASH_SIZE = 8                            # 8x8 = 64 bits per hash
NUM_HASHES = 6                           # hashes per image
BATCH_SIZE = 64                          # files per worker task

# Popcount lookup table: set-bit count for every uint8 value 0-255
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def compute_hashes_batch(filepaths):
    """
    Hash a batch of images in one worker call (reduces IPC overhead).
    Uses dhash (faster than phash) and packs bits at hash time.
    Returns packed (6, 8) uint8 arrays - no conversion needed later.

    6 hashes per image to catch asymmetric crops:
      0: full image
      1: center 60%   (symmetric crop)
      2: top 80%      (catches images trimmed from the bottom)
      3: bottom 80%   (catches images trimmed from the top)
      4: left 80%     (catches images trimmed from the right)
      5: right 80%    (catches images trimmed from the left)
    """
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
                    img.crop((int(w*.2), int(h*.2), int(w*.8), int(h*.8))),  # center 60%
                    img.crop((0,         0,          w,         int(h*.8))),  # top 80%
                    img.crop((0,         int(h*.2),  w,         h        )),  # bottom 80%
                    img.crop((0,         0,          int(w*.8), h        )),  # left 80%
                    img.crop((int(w*.2), 0,          w,         h        )),  # right 80%
                ]
                packed = np.array([
                    np.packbits(imagehash.dhash(c, hash_size=HASH_SIZE).hash.flatten())
                    for c in crops
                ], dtype=np.uint8)  # (6, 8)
                results.append((filepath, packed))
        except Exception:
            results.append((filepath, None))
    return results


def load_cache(resume):
    """Load binary .npz cache. Returns (path_list, hash_array (n,6,8)) or empty."""
    if resume and os.path.exists(CACHE_FILE):
        try:
            print(f"\nLoading cache from '{CACHE_FILE}'...")
            data = np.load(CACHE_FILE, allow_pickle=True)
            paths = data['paths'].tolist()
            hashes = data['hashes']  # (n, 6, 8)
            if hashes.ndim == 3 and hashes.shape[1] == NUM_HASHES:
                print(f"  {len(paths)} hashes loaded from cache")
                return paths, hashes
        except Exception:
            print("  Cache unreadable - starting fresh")
    return [], np.empty((0, NUM_HASHES, 8), dtype=np.uint8)


def save_cache(paths, hashes):
    np.savez_compressed(CACHE_FILE,
                        paths=np.array(paths, dtype=object),
                        hashes=hashes)
    print(f"Cache saved to '{CACHE_FILE}'")


def find_similar_photos(root_dir, similarity_threshold=10, output_csv="similar_photos.csv",
                        workers=None, resume=False):

    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {workers} worker processes")

    ### STEP 1: Scan ###############
    print("\nScanning for image files...")
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                all_files.append(os.path.join(dirpath, filename))
    print(f"Found {len(all_files)} image files")

    ### STEP 2: Load cache ###############
    cached_paths, cached_hashes = load_cache(resume)
    cached_set = set(cached_paths)

    to_hash = [f for f in all_files if f not in cached_set]
    print(f"{len(to_hash)} files to hash  |  {len(cached_set)} already cached")

    ### STEP 3: Hash in parallel batches ###############
    new_paths, new_hashes_list = [], []

    if to_hash:
        batches = [to_hash[i:i + BATCH_SIZE] for i in range(0, len(to_hash), BATCH_SIZE)]
        errors = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compute_hashes_batch, b): b for b in batches}
            with tqdm(total=len(to_hash), desc="Hashing images", unit="img") as pbar:
                for future in as_completed(futures):
                    for filepath, packed in future.result():
                        if packed is not None:
                            new_paths.append(filepath)
                            new_hashes_list.append(packed)
                        else:
                            errors += 1
                        pbar.update(1)

        if errors:
            print(f"  {errors} file(s) skipped due to errors")

        new_hashes = np.stack(new_hashes_list)                              # (new_n, 6, 8)
        all_paths  = cached_paths + new_paths
        all_hashes = np.concatenate([cached_hashes, new_hashes], axis=0)   # handles empty cache
        save_cache(all_paths, all_hashes)
    else:
        all_paths, all_hashes = cached_paths, cached_hashes

    ### STEP 4: Build hash matrix ###############
    path_to_idx   = {p: i for i, p in enumerate(all_paths)}
    valid_files   = [f for f in all_files if f in path_to_idx]
    stacked       = all_hashes[[path_to_idx[f] for f in valid_files]]  # (n, 6, 8)
    n             = len(valid_files)
    print(f"\nHash matrix: {n} images × {NUM_HASHES} hashes × 8 bytes "
          f"({stacked.nbytes / 1024**2:.1f} MB)")

    ### STEP 5: Vectorized comparison (6 iterations instead of 36) ###############
    # For each i-hash type, compare against ALL 6 j-hashes at once by flattening j.
    # This replaces 36 Python loop iterations with 6 larger numpy operations.
    print("Comparing hashes...")
    similar_photos = []
    chunk_size     = 1000  # per-iteration memory: chunk * chunk*6 * 8 bytes = ~48 MB

    total_chunks = (n + chunk_size - 1) // chunk_size
    with tqdm(total=total_chunks, desc="Comparing", unit="chunk") as pbar:
        for i in range(0, n, chunk_size):
            si = stacked[i:i + chunk_size]          # (m, 6, 8)
            for j in range(i, n, chunk_size):
                sj      = stacked[j:j + chunk_size] # (k, 6, 8)
                m, k    = len(si), len(sj)
                sj_flat = sj.reshape(-1, 8)          # (k*6, 8)

                similar_mask = np.zeros((m, k), dtype=bool)
                for ki in range(NUM_HASHES):
                    ci   = si[:, ki, :]              # (m, 8) - one hash type for all i images
                    xor  = ci[:, np.newaxis, :] ^ sj_flat[np.newaxis, :, :]  # (m, k*6, 8)
                    dists = (_POPCOUNT[xor]
                             .sum(axis=2)
                             .reshape(m, k, NUM_HASHES))                      # (m, k, 6)
                    similar_mask |= (dists <= similarity_threshold).any(axis=2)

                rows, cols = np.where(similar_mask)
                for r, c in zip(rows, cols):
                    gi, gj = i + r, j + c
                    if gi < gj:
                        similar_photos.append((valid_files[gi], valid_files[gj]))
            pbar.update(1)

    # Deduplicate - multiple hash combinations can flag the same pair independently
    similar_photos = list(dict.fromkeys(similar_photos))

    ### STEP 6: Write CSV ###############
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image 1", "Image 2"])
        writer.writerows(similar_photos)

    print(f"\nDone! Found {len(similar_photos)} similar pair(s).")
    print(f"Results written to '{output_csv}'")
    return similar_photos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find similar/duplicate photos in nested folders and write results to CSV."
    )
    parser.add_argument("root_dir",               help="Directory to search recursively")
    parser.add_argument("--threshold", type=int,  default=10,
                        help="Hash difference threshold 0-64 (default: 10). Lower = stricter. Suggested range: 8-15")
    parser.add_argument("--output",               default=os.path.join(SCRIPT_DIR, "similar_photos.csv"),
                        help="Output CSV file path (default: same folder as script)")
    parser.add_argument("--workers",  type=int,   default=None,
                        help="Parallel worker count (default: CPU cores - 1)")
    parser.add_argument("--resume",   action="store_true",
                        help="Reuse existing photo_hashes_cache.npz to skip already-hashed files")

    args = parser.parse_args()
    find_similar_photos(args.root_dir, args.threshold, args.output, args.workers, args.resume)
