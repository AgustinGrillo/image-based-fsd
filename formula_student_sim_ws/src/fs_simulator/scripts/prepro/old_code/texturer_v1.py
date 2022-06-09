

"""
Change Texture from objects
"""
def change_texture(path, texture_num):
    # read input file
    fin = open(path, "rt")
    fin.seek(0)
    for line in fin:
        if line.find('_texture') is not -1:
            init_pos = line.find('_texture')
            end_pos = line.find('.jpg')
            str2replace = line[init_pos-1: end_pos+4]
            str_start = line[init_pos-1: init_pos+8]
            str_end = line[end_pos: end_pos+4]

    replaceby = str_start+str(texture_num)+str_end
    fin.seek(0)

    # read file contents to string
    data = fin.read()
    # replace all occurrences of the required string
    data = data.replace(str2replace, replaceby)
    # close the input file
    fin.close()
    # open the input file in write mode
    fin = open(path, "wt")
    # overrite the input file with the resulting data
    fin.write(data)
    # Returan cursor to the beggining
    fin.seek(0)
    # close the file
    fin.close()
