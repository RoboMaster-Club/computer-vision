import os
import sys
import random

WIDTH = 856
HEIGHT = 480

def parseArgv():
    args = dict()
    for i in range(len(sys.argv[1:]) // 2):
        if sys.argv[1 + 2 * i] == '-i' or sys.argv[1 + 2 * i] == 'input':
            args['input'] = sys.argv[2 + 2 * i]
        elif sys.argv[1 + 2 * i] == 'image':
            args['image'] = sys.argv[2 + 2 * i]
        elif sys.argv[1 + 2 * i] == 'folder':
            argv['output'] = sys.argv[2 + 2 * i]
        elif sys.argv[1 + 2 * i] == 'train':
            args['train'] = sys.argv[2 + 2 * i]
        elif sys.argv[1 + 2 * i] == 'test':
            args['test'] = sys.argv[2 + 2 * i]
        elif sys.argv[1 + 2 * i] == 'data':
            args['data'] = sys.argv[2 + 2 * i]
        elif sys.argv[1 + 2 * i] == 'names' or sys.argv[1 + 2 * i] == 'name':
            args['name'] = sys.argv[2 + 2 * i]
    return args


def readLabels(file):
    labels = dict()
    with open(file, "r") as f:
        raw_data = f.read().split('\n')
    for row in raw_data[:-1]:
        row = row.split(' ')
        key = row[0].split('/')[-1]
        val = []
        for i in range(int(row[1])):
            val.append([int(ii) for ii in row[2+4*i: 6+4*i]])
            val[i][0] /= WIDTH
            val[i][1] /= HEIGHT
            val[i][2] /= WIDTH
            val[i][3] /= HEIGHT
        labels[key] = val
    return labels


def generateTxtLabels(labels, folder, test=0.2):
    trainingList = []
    testingList = []
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    for f in files:
        if f in labels and len(labels[f]) > 0:
            with open(os.path.join(folder, f.replace('.jpg', '.txt')), 'w') as fp:
                for lbl in labels[f]:
                    fp.write(f'1 {lbl[0]} {lbl[1]} {lbl[2]} {lbl[3]}\n')
                choose = random.uniform(0, 1)
                if choose > test:
                    trainingList.append(os.path.abspath(os.path.join(folder, f)))
                else:
                    testingList.append(os.path.abspath(os.path.join(folder, f)))

    return trainingList, testingList


def generateListFile(path, imageList):
    with open(path, 'a') as f:
        for img in imageList:
            f.write(f'{img}\n')


def generateName(path):
    with open(path, 'w') as f:
        f.write('Robot')


def generateData(path, test, train, name):
    with open(path, 'w') as f:
        f.write('classes=1\n')
        f.write(f'train={os.path.abspath(train)}\n')
        f.write(f'valid={os.path.abspath(test)}\n')
        f.write(f'names={os.path.abspath(name)}\n')
        f.write('backup=./backup')


if __name__ == '__main__':
    args = parseArgv()
    if 'input' in args and 'image' in args and 'train' in args and 'test' in args:
        labels = readLabels(args['input'])
        training_set, testing_set = generateTxtLabels(labels, args['image'], test=0.2)

        generateListFile(args['train'], training_set)
        generateListFile(args['test'], testing_set)

        generateName(args['name'])
        generateData(args['data'], args['test'], args['train'], args['name'])

    else:
        sys.stderr.write('Missing argument(s)')
        sys.stderr.write('Please have input, image, train, test, data, name arguments')
        exit(1)







