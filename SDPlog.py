import sys


# 主要用于记录控制台打印输出

class Logger(object):
    def __init__(self, filename="aaa.txt",
                 stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
