from sklearn.ensemble import RandomForestClassifier
import math
import re

domainlist = []

class Domain:
    def __init__(self, nam, lab, _len, ent, num, seg):
        self.name = nam
        self.label = lab
        self.length = _len
        self.entropy = ent
        self.number = num
        self.segment = seg

    def return_data(self):
        return [self.length, self.entropy, self.number, self.segment]

    def return_label(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def cal_entropy(string):
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    return - sum([p * math.log(p) / math.log(2.0) for p in prob]) # 熵值

def entropy_ideal(length):
    return -1.0 * math.log(1.0 / length) / math.log(2.0) # 理想熵

def init_data(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip() # 去除首尾空格
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length = len(line)
            entropy = cal_entropy(line)
            number = len(re.findall(r'\d', line))
            segment = len(tokens[0].split("."))
            domainlist.append(Domain(name, label, length, entropy, number, segment))
    f.close()

def train(clf):
    # Initialize Raw Objects
    init_data("train.txt")
    featureMatrix = []
    labelList = []
    # Initialize Matrix
    for item in domainlist:
        featureMatrix.append(item.return_data())
        labelList.append(item.return_label())
    print("Begin Training")
    clf.fit(featureMatrix, labelList)

def predict(filename, clf):
    print("Begin Predicting")
    with open("result.txt", 'w') as f_w:
        with open(filename) as f_r:
            for line in f_r:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                length = len(line)
                entropy = cal_entropy(line)
                number = len(re.findall(r'\d', line))
                segment = len(line.split("."))
                str=line
                if clf.predict([[length, entropy, number, segment]])==0:
                    str+=",notdga\n"
                else:
                    str+=",dga\n"
                f_w.write(str)
        f_r.close()
    f_w.close()

clf = RandomForestClassifier(random_state=0)
train(clf)
predict("test.txt", clf)
