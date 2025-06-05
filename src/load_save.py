import os

#  Helperi
def save_list_to_txt(lst, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in lst:
            f.write(f"{item}\n")

def load_list_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
