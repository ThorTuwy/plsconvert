from tqdm import tqdm

class ProgressBar:
    def __init__(self, total: int, bar_format: str = '|{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]'):
        self.total = total
        self.bar_format = bar_format
        self.pbar = tqdm(total=total, bar_format=bar_format)

    def update(self, n: int = 1):
        self.pbar.update(n)

    def extend(self, n: int):
        self.pbar.total += n

    def close(self):
        self.pbar.close()
