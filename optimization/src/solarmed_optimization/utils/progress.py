import time
from tqdm import tqdm

def update_bar_every(pbar: tqdm, interval=0.5) -> None:
    """ Updates progress bar every `interval` seconds """
    while True:
        time.sleep(interval)
        pbar.refresh()