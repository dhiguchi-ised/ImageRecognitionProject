from PIL import Image
import os

PATH = "C:\\Users\\Higuchid\\Documents\\Python Scripts\\TensorFlow CNN\\data\\IPC_ORGANIZATION\\Class D - Textiles; Paper"

# Takes in a file path of an image, filePath, and also a string containing
#   just that file's name for convenience. Will then split the file specified
#   into individual parts, and save them with an indicator specifying their
#   page in the multi-image file they are pulled from.
# Requires: filePath is a multi-image tif file
def parse_tif(filePath, filename):
    with Image.open(filePath) as img:

        numFramesPerTif = img.n_frames
        for i in range (numFramesPerTif):
            img_name = filename[:-4] + '_Page_%s.tif'%(i,)
            try:
                img.seek(i)
                img.save(os.path.join(PATH, img_name))
            except EOFError: 
                print("end of file error")
    img.close()
    os.remove(filePath) # removes multipage file since it has been split

# will apply function to entire directory, but only to the multipage tif files,
#   distinguished by having 'DRAWINGS' in their file name
for img in os.listdir(PATH):
    if 'DRAWINGS' in img:
        parse_tif(os.path.join(PATH, img), img)

