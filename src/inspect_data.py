import os

DATA_DIR = "data/TCGA-PANCAN-HiSeq-801x20531"

print("Files inside dataset folder:")
print("-" * 50)

for file in os.listdir(DATA_DIR):
    print(file)