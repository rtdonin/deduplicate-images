# deduplicate-images

A small project in Python to remove duplicate images.

## Reason for this project

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

Note: As of this update, testing and general optimization are still in progress to further improve the performance and user experience of the application.
