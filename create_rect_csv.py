import csv
from pathlib import Path
from random import random

output = Path("output")
output.mkdir(exist_ok=True)

rectangles = [(i, random() * 100, random() * 100) for i in range(10000)]

with open(output / "rectangles.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("id", "width", "length"))
    writer.writerows(rectangles)
