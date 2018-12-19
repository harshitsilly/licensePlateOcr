from PIL import Image
import os

aImageFiles = os.listdir("images")
for imageFile in aImageFiles:
    img = Image.open("images\\" + imageFile)
    basewidth = 152 
    hsize = 34
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save("data\\img\\" + imageFile)
