#!/usr/bin/env python
import rospy
import sensor_msgs.msg
import rospkg
import os

rospy.init_node('test')

# initialize ROS publisher
pub = rospy.Publisher('/joint_states',
                      sensor_msgs.msg.JointState, queue_size=10)

# wait to establish connection between the controller
while pub.get_num_connections() == 0:
    rospy.sleep(0.1)

# fill ROS message
state = sensor_msgs.msg.JointState()
state.name = ["shoulder_pan_joint", "shoulder_lift_joint",
                    "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "box_x", "box_y", "box_rotate"]
tstep = 0.5
rate = rospy.Rate(10.0)

rospack = rospkg.RosPack()


with open(os.path.join(rospack.get_path('trajectory_my'),'logs','urtrajectory.txt'), "r") as fin:
    #next(fin)
    for line in fin:
        line = line[:-2]
        t = [float(e) for e in line.split(",")]
        state.position = [t[1],t[2],t[3],t[4],t[5],t[6], t[7], t[8], t[9]]
	state.velocity = [t[10],t[11],t[12],t[13],t[14],t[15], t[16], t[17], t[18]]
        state.header.stamp = rospy.Time.now()
	pub.publish(state)
	rospy.sleep(tstep)

