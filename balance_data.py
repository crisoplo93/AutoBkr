import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

file_name = 'training_data_OTAI.npy'

train_data = np.load(file_name)
print(len(train_data))

df = pd.DataFrame(train_data)
print(Counter(df[1].apply(str)))

up = []
down = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0]:
        up.append([img, choice])

    elif choice == [0, 1]:
        down.append([img, choice])
up = up[:len(down)]
down = down[:len(down)]

final_data = up + down
shuffle(final_data)

print(len(final_data))

np.save('training_data_OTAIv2.npy', final_data)
np.load = np_load_old