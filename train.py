
from dataloader import SongData
from torch.utils.data import DataLoader
import sys

if __name__ == '__main__':

    print("Creating Data Set \n")
    sys.stdout.flush()
    songSet = SongData('.', 'songdata.csv')

    print("Making loader \n")
    sys.stdout.flush()
    loader = DataLoader(songSet, shuffle=True, batch_size=128)

    print("Beginning iteration")
    sys.stdout.flush()
    for idx, batch in enumerate(loader):
        print(batch)
