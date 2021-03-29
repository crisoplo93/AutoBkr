import numpy as np
from alexnet import alexnet
import time
WIDTH = 110
HEIGHT = 58
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'autobrk-{}-{}-{}-epochs-3K-data.model'.format(LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

hm_data = 22
for i in range(EPOCHS):
    train_data = np.load('training_data_OTAIv2.npy')

    train = train_data[:-100]
    test = train_data[-100:]

    X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)
    time.sleep(15)

np.load = np_load_old
# tensorboard --logdir=foo:C:\Users\Cris_\PycharmProjects\AutoBrk\log