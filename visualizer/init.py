#!/usr/bin/env python
import rospy
import sensor_msgs.msg

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
                    "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "box_x", 		            "box_y", "box_rotate"]

state.position = [0,-1,1.86,-0.8,1.57,0,0.49, 0.131,0]
state.velocity = [0,0,0,0,0,0,0,0,0]
state.effort = [0,0,0,0,0,0,0,0,0]

while not rospy.is_shutdown():
  state.header.stamp = rospy.Time.now()
  pub.publish(state)
  rospy.sleep(0.1)

