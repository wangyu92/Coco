class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'w')
        
    def log(self, msg):
        self.file.write(msg + '\n')