import glob

class MultipleFileReader:

    def readFilesFromDirectory(self, path):
        return self.readFiles(glob.glob(path + "/*.txt"))
    
    def readFiles(self, files):
        document = ""
        for file in files:
            with open(file,'r') as content_file:
                document += " " + content_file.read()
        return document


print(MultipleFileReader().readFilesFromDirectory("tech"))