from os import listdir, makedirs
from os.path import isdir
#Place in labels folder

output = "output/"
splits = [
    "train/",
    "test/",
    "val/",
]
separator = ' '

for folder in splits:
    filenames = listdir(folder)

    if not isdir(output+folder): makedirs(output+folder)
    for filename in filenames:
        with open(folder+filename, 'r') as file:
            with open(output+folder+filename, "w") as o:
                for line in file:
                    values = line.split(separator)
                    #if values[0] == "41": continue #SHOULD skip traffic signals(this is untested)
                    values[0] = '0'
                    lineOut = separator.join(values)
                    o.write(lineOut)