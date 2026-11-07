
from pathlib import Path

DATA_PATH = Path("data/raw/MI.data")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}\n"
        "Place MI.data inside data/raw/"
    )

print("=" * 60)
print("DATASET FILE INFORMATION")
print("=" * 60)

print(f"File: {DATA_PATH.resolve()}")
print(f"Size: {DATA_PATH.stat().st_size / 1024:.2f} KB")

print("\n" + "=" * 60)
print("FIRST 10 LINES")
print("=" * 60)

with DATA_PATH.open("r", encoding="utf-8-sig", errors="replace") as file:
    for i, line in enumerate(file):
        if i == 10:
            break
        print(f"{i + 1}: {line.rstrip()}")
