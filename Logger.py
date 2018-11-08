import csv
import os
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, arg, save_path):
        self.log_path = save_path + "/log.txt"
        self.lines= []
        if os.path.exists(self.log_path) is False:
            with open(self.log_path, "a+", encoding="utf8") as f:
                f.write(str(arg)+"\n")
                f.write("args end\n")
        

    def will_write(self, line):
        print(line)
        self.lines.append(line)

    def flush(self):
        with open(self.log_path, "a", encoding="utf8") as f:
            for line in self.lines:
                f.write(line + "\n")
        self.lines = []

    def write(self, line):
        self.will_write(line)
        self.flush()

    def read(self):
        train, val, test = [], [], []

        with open(self.log_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                line = line.split()
                if line[0] == "[Train]":
                    epoch = int(line[1][7:])
                    loss = float(line[2][6:])
                    train.append((epoch, loss))
                elif line[0] == "[Val]":
                    epoch = int(line[1][7:])
                    jss = float(line[2][6:])
                    val.append((epoch, jss))
                elif line[0] == "[Test]":
                    epoch = int(line[1][7:])
                    jss = float(line[2][6:])
                    test.append((epoch, jss, dice))
                elif line[0] == "Total":
                    self.best_jss = float(line[1][6:])
                    self.best_dice = float(line[2][7:])

        self.train = train
        self.val = val
        self.test = test
