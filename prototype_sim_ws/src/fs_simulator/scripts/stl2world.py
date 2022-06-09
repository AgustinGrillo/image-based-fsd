#!/usr/bin/env python
import rospy
import math
import random
def parser():

    fi = open('Nurburgring_yellow.stl', 'r')
    fo = open('Nurburgring_yellow.world', 'w')

    line = fi.readline()
    cnt = 1
    data = []

    while line:
        line = fi.readline()
        if line.find('vertex ') != -1 and len(line)>10: #parse
            data.append(line.strip("vertex ")[:-10])
        cnt += 1

    print (len(data))
    data = list( dict.fromkeys(data)) #delete exact duplicates
    print (len(data))

    cnt = 1
    for line in data:
        fo.write("         <link name='yellow_cone_"+str(cnt)+"'>\n")
        fo.write("           <pose frame=''> "+ line +" -0.005 0 -0 "+str(random.randint(0, 20))+"</pose>\n")
        fo.write("          <collision name=\"collision\"><geometry><box><size> 0.2 0.2 0.01</size>\n")
        fo.write("            </box></geometry></collision>\n")
        fo.write("           <visual name=\"visual\"><geometry><mesh>\n")
        fo.write("                 <uri>model://YellowCone.dae</uri>\n")
        fo.write("           </mesh></geometry></visual>\n")
        fo.write("         </link>\n")
        fo.write(" \n")
        cnt += 1

    fi.close()
    fo.close()


if __name__ == '__main__':
    parser()
