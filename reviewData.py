import numpy as np
import cv2
import time

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

i = 1900

file_name = 'training_data_OTAI.npy'
train_data = np.load('training_data_OTAI.npy')
# train_data = np.delete(train_data, 1915)
#
# np.save(file_name, train_data)
print(len(train_data))

for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test', img)
    print(choice)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        np.load = np_load_old
        break
    np.load = np_load_old
np.load = np_load_old

