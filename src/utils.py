import numpy as np
l


def data_generator(batch_size, data):
    x, y = data
    num_data = x.shape[0]
    batches = num_data // batch_size

    data = list(zip(x, y))
    np.random.shuffle(data)
    x, y = zip(*data)

    for i in range(batches):
        x_batch = x[i * batch_size : (1 + i) * batch_size]
        y_batch = y[i * batch_size : (i + 1) * batch_size]
        yield x_batch, y_batch





if __name__ == '__main__':
    data = ([1,2,3,4,5],[6,7,8,9,0])
    print(data_generator())
