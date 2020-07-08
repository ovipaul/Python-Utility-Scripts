import glob, os

fSave = open('Train.txt', 'w')

for infile in glob.glob("*.png"):
    fName, ext = os.path.splitext(infile)
    fSave.write(fName+'\n') 
fSave.close() 