import random


class UseData:

    def __init__(self):
        self.data = []

    def extract_data(self, pathToData):
        with open(pathToData, 'r') as file:
            data = file.read()
            data = data.split('\n')
            for i in range(1, len(data)):
                if data[i].split(' ')[0] != "@attribute":
                    data = data[(i + 3):]
                    break
            for i in range(len(data)):
                row = [x.strip() for x in data[i].split(",")]
                if row != [] or row != [""]:
                    self.data.append(row)  # add row in data

    def cross_validation(self):
        trainingSample = []
        testSample = []
        random.shuffle(self.data)
        partLen = int(len(self.data)/10)
        dataSplit = [self.data[partLen*k:partLen*(k+1)] for k in range(10)]
        for i in range(len(dataSplit)):
            testSample.append(dataSplit[i])
            sumTraining = []
            for j in range(len(dataSplit)):
                if j != i:
                    sumTraining += dataSplit[j]
            trainingSample.append(sumTraining)
        return trainingSample, testSample

    def get_data(self):
        return self.data
