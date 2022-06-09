
import os


dir_path = '/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/meshes/asphalt/materials/textures/tester/'

base_name = 'floor_texture'


# Function to rename multiple files
def main():
    for count, filename in enumerate(os.listdir(dir_path)):
        dst = base_name + str(count+1) + ".jpg"
        if os.path.exists(dir_path + dst):
            print('[WARNING] Aborting to avoid deletion of files')
            print('First rename files with a different name than the actual one, and then rename using the desired final name.')
            break

        src = dir_path + filename
        dst = dir_path + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
