## deduplicate-images

A small project in Python to remove duplicate images.

# Reason for this project

As a software developer, I wanted to exercise my Python skills and create a solution for removing duplicate images. Rather than relying on existing software, I decided to write my own tool to gain hands-on experience and tailor it to my specific needs.

Below are the steps I took to make this small project:

<b>Step 1:</b> Find a way to identify duplicates regardless of cropping.

<b>Step 2:</b> Make sure that we don't have A, B and B, A (the same pair but in a different order).

<b>Step 3:</b> Let the user as the human go through the duplications.

<b>Step 4:</b> Create user-friendly functionality (saving, going back).

<b>Step 5:</b> Optimization of hashing images.

<b>Step 6:</b> Create an exe file to make this user-friendly and not require a CLI. This needs to be something that someone without a technical background can download and use.

<b>Step 7:</b> Continuous testing and general optimization (ongoing as of February 27, 2026).

## Development Process

1. **Basic script** - Single-threaded script using average_hash and a Python loop to compare all pairs. Functional but slow for large collections.

2. **Parallel hashing** - Switched to ProcessPoolExecutor to hash images across all CPU cores simultaneously. Reduced hashing time significantly.

3. **Vectorized comparison** - Replaced the nested Python comparison loop with NumPy broadcasting. Eliminated the O(n²) Python bottleneck.

4. **Packed bit hashes** - Stored hashes as packed uint8 arrays (8 bytes each) instead of 64-element boolean arrays. 8x memory reduction, faster XOR-based Hamming distance via popcount lookup table.

5. **Crop detection** - Switched from average_hash to dhash. Added 6 hashes per image (full + 5 directional crops) and compare all 36 combinations per pair. Catches images trimmed from any edge.

6. **Batch processing** - Grouped files into batches of 64 before submitting to workers. Reduced inter-process communication overhead.

7. **Binary cache** - Replaced JSON cache with compressed NumPy .npz format. Much faster to load on resume runs.

8. **Deduplication** - Added deduplication of output pairs since multiple hash combinations could independently flag the same pair.

9. **Review GUI** - Built review_duplicates.py using tkinter. Side-by-side image display, keyboard controls, undo (back button), and session resume via progress file.

10. **GUI app** - Combined both scripts into a single standalone application with a home screen, Find/Review/Remove screens, background threading with live progress, and startup logic tests. Packaged as a .exe using PyInstaller.

Note: As of February 27, 2026, testing and general optimization are still in progress to further improve the performance and user experience of the application.

## Requirements

Python 3.8 or higher.

```
python -m pip install Pillow imagehash numpy tqdm
```

## Script 1 - Find Duplicates

Scans a folder recursively, computes perceptual hashes for every image, and finds similar pairs. Results are saved to `similar_photos.csv` in the same folder as the script.

Basic usage:

```
python find_similar_photos.py "C:\path\to\your\photos"
```

With options:

```
python find_similar_photos.py "C:\Photos" --threshold 8 --output "C:\Photos\results.csv" --workers 4 --resume
```

Options:
| Flag | Description | Default |
| --- | --- | --- |
| `--threshold` | Similarity cutoff (0-64, lower = stricter match) | `10` |
| `--output` | Path for the output CSV file | Same folder as script |
| `--workers` | Number of CPU cores to use | All cores minus one |
| `--resume` | Reuse cached hashes from a previous run (skips already-hashed files) | Off |

Output files:
* `similar_photos.csv` - pairs of similar images
* `photo_hashes_cache.npz` - hash cache (used when `--resume` is passed on a future run)

## Script 2 - Review Duplicates

Opens `similar_photos.csv` and lets you compare each pair side-by-side in a window. You decide which image to keep and which to remove.

Usage:

```
python review_duplicates.py
```

The script reads `similar_photos.csv` from the same folder automatically. To specify a different CSV:

```
python review_duplicates.py "C:\path\to\results.csv"
```

Keyboard controls:
| Key | Action |
| --- | --- |
| `1` | Keep left image, move right to removed folder |
| `2` | Keep right image, move left to removed folder |
| `3` | Skip - images are not actually duplicates |
| `B` | Go back to previous pair (undoes the last move) |
| `Esc` | Quit and save progress |

Notes:
* Removed files are moved to `duplicates_removed` on your Desktop - they are not permanently deleted.
* Progress is saved automatically after every decision. If you quit mid-way through, the next run will ask if you want to resume where you left off.
* Pairs where one image was already removed by a previous decision are skipped automatically.