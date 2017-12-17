class MLModel:
    def __init__(self,data):
        '''

        :param data: data object, defined in ReadData as FetchingData
        '''
        self.data=data
        self.name=''
        self.model=None
    def train(self):
        pass
    def predict(self,x):
        '''

        :param x: an array of input vectors
        :return: an array of preidtcted labels
        '''
        pass
    def save(self):
        pass
    def load(self):
        pass