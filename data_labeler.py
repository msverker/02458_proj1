import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- EDIT THESE PATHS ---
IMAGE_FOLDER = Path(r"C:\Users\richa\OneDrive\Documents\University\Cognitive Modelling\subset\subset")
OUTPUT_CSV   = Path(r"C:\Users\richa\OneDrive\Documents\University\Cognitive Modelling\subset\labels.csv")
# -------------------------

VALID_LABELS = {"1", "2", "3", "4", "5"}

def prompt_label(filename: str) -> int:
    while True:
        val = input(f"Label for '{filename}' (1, 2, 3, 4, 5) [q to quit]: ").strip()
        if val.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if val in VALID_LABELS:
            return int(val)
        print("Invalid input. Please enter 1, 2, 3, 4, or 5 (or 'q' to quit).")

def load_existing_labels(csv_path: Path) -> dict:
    """Return dict of {filename: label} if CSV exists, else empty dict."""
    labels = {}
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    labels[row[0]] = row[1]
    return labels

def main():
    images = sorted(
        [p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}]
    )
    if not images:
        print("No .jpg/.jpeg images found.")
        return

    existing = load_existing_labels(OUTPUT_CSV)
    print(f"Found {len(images)} images. {len(existing)} already labeled; skipping those.")

    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header if file is empty
        if OUTPUT_CSV.stat().st_size == 0:
            writer.writerow(["filename", "label"])

        try:
            for img_path in images:
                if img_path.name in existing:
                    continue

                # Show image with matplotlib
                img = mpimg.imread(img_path)
                plt.imshow(img)
                plt.axis("off")
                plt.title(img_path.name)
                plt.show(block=False)

                label = prompt_label(img_path.name)

                plt.close()  # close the window after rating

                writer.writerow([img_path.name, label])
                print(f"Saved: {img_path.name},{label}")

        except KeyboardInterrupt:
            print("\nExiting. Progress saved.")

if __name__ == "__main__":
    main()
