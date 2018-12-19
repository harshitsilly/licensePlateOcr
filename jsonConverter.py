import os
import json

aImageFiles = os.listdir("images")
for imageFile in aImageFiles:
    data={"tags": ["train", "train"], "description": imageFile.split('.')[0], "objects": [], "size": {"height": 34, "width": 152}}
    with open("data\\ann\\" + imageFile.split('.')[0] + ".json", 'w') as fp:
        json.dump(data, fp)