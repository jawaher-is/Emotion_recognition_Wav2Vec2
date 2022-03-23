import numpy as np

def nested_array_catcher(array, catch_type=np.ndarray, target_type=np.float32):
    for i in range(len(array)):
        if isinstance(array[i], target_type):
            continue
        else:
            print(8, type(array[i]), array[i]) # <class 'numpy.ndarray'>
            # if isinstance(array[i], catch_type):
            print(array.shape, array)
            array = array[i]
            print('new type: ', type(array))
            print(array.shape, array)
            print(type(array[i]), array[i])
            return array
    return array
