# file = open('/home/agus/Documents/Proyectos/fs_itba/src/fs_simulator/scripts/prepro/poses.txt', 'r')


def get_pose(poses_file, pose_num):

    position = []
    orientation = []

    poses_file.seek(0)
    for line in poses_file:
        if line.find('#Mid ' + str(pose_num) + ':') > -1:
            pos = poses_file.next()
            while pos.find('position') is -1:
                pos = poses_file.next()
            x = poses_file.next()
            position.append( float(x[x.find('x:') + 3:-1]) )
            y = poses_file.next()
            position.append(float(y[y.find('y:') + 3:-1]))
            z = poses_file.next()
            position.append(float(z[z.find('z:') + 3:-1]))

            orn = poses_file.next()
            while orn.find('orientation') is -1:
                orn = poses_file.next()
            x = poses_file.next()
            orientation.append(float(x[x.find('x:') + 3:-1]))
            y = poses_file.next()
            orientation.append(float(y[y.find('y:') + 3:-1]))
            z = poses_file.next()
            orientation.append(float(z[z.find('z:') + 3:-1]))
            w = poses_file.next()
            orientation.append(float(w[w.find('w:') + 3:-1]))

    poses_file.seek(0)  # move cursor again to beggening of file for re-reading availability
    return position, orientation


def get_num_poses(poses_file):
    counter = 0
    poses_file.seek(0)
    for file_line in poses_file:
        if file_line.find('#Mid') > -1:
            counter += 1

    poses_file.seek(0)
    return counter
