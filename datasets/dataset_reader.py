'''
Author: Cristiano-3 chunanluo@126.com
Date: 2022-11-30 11:19:02
LastEditors: Cristiano-3 chunanluo@126.com
LastEditTime: 2022-12-02 10:40:09
FilePath: /dbnet_plus/datasets/dataset_reader.py
Description: Dataset Loader
'''


class DatasetReader(object):
    """implementation of data pipeline
    """
    def __init__(self, params):
        pass
    
    def __call__(self):

        def generator():
            i = 0
            while True:
                i+=1
                yield i
            
        return generator()

if __name__ == "__main__":
    reader = DatasetReader([])
    generator = reader()
    for i in range(20):
        d = next(generator)
        print(d, end=' ')
