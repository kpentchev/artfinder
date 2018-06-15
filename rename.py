import glob, os, re

dir = "/Users/kpentchev/artmimir/test_data_raw/"
picture = "creation_of_adam"
pattern = re.compile(r"(.*)\.(jpg|jpeg|png)")
newTitle = picture + "_test_%s.jpg"

gl = glob.glob(os.path.join(dir + picture, "*.*"))

i = 1
for pathAndFilename in gl:
    if pattern.search(pathAndFilename):
        #print(pathAndFilename)
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, os.path.join(dir + picture, newTitle % i))
        i += 1