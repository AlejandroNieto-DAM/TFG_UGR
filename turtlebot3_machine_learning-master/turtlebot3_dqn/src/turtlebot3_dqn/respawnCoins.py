#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelConfiguration
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState

from geometry_msgs.msg import Pose
from std_msgs.msg import String
from gazebo_msgs.srv import SetModelState
import math
import time
from threading import Thread, Event
from math import pi

class RespawnCoin():
    def __init__(self, id):
        
        self.modelPath = "/home/nietoff/tfg/src/turtlebot3_simulations-master/turtlebot3_gazebo/models/turtlebot3_coin/coin/model.sdf"

        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')

        self.coin_position = Pose()
        self.init_coin_x = 0.6
        self.init_coin_y = 0.0
        self.init_coin_z = 0.1
        self.coin_position.position.x = self.init_coin_x
        self.coin_position.position.y = self.init_coin_y
        self.coin_position.position.z = self.init_coin_z
        #self.coin_position.orientation.y = 1.57

        self.id = str(id)
        self.modelName = 'coin_' + self.id
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_coin_x = self.init_coin_x
        self.last_coin_y = self.init_coin_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)

        self.check_model = False
        self.index = 0

        self.stop_event = Event()

        

    def spin_move(self):

        rospy.wait_for_service('/gazebo/set_model_state')
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        
        angle = 0
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            angle += 0.01
            if angle > 2 * math.pi:
                angle -= 2 * math.pi

            model_state = ModelState()
            model_state.model_name = self.modelName
            model_state.pose.position.x = self.coin_position.position.x if self.coin_position else 0.0
            model_state.pose.position.y = self.coin_position.position.y if self.coin_position else 0.0
            model_state.pose.position.z = self.coin_position.position.z if self.coin_position else 0.0

            # Convert the angle to quaternion
            model_state.pose.orientation.z = math.sin(angle / 2.0)
            model_state.pose.orientation.w = math.cos(angle / 2.0)

            model_state.reference_frame = 'world'

            try:
                set_model_state(model_state)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)

            
    def start_spin_move_thread(self):
        spin_thread = Thread(target=self.spin_move)
        spin_thread.start()
        self.spin_thread = spin_thread

    def stop_spin_move_thread(self):
        self.stop_event.set()

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "coin_" + self.id:
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.coin_position, "world")
                rospy.loginfo("Coin position : %.1f, %.1f", self.coin_position.position.x,
                              self.coin_position.position.y)
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if self.stage != 4:
            while position_check:
                coin_x = random.randrange(-12, 13) / 10.0
                coin_y = random.randrange(-12, 13) / 10.0
                if abs(coin_x - self.obstacle_1[0]) <= 0.4 and abs(coin_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(coin_x - self.obstacle_2[0]) <= 0.4 and abs(coin_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(coin_x - self.obstacle_3[0]) <= 0.4 and abs(coin_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(coin_x - self.obstacle_4[0]) <= 0.4 and abs(coin_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(coin_x - 0.0) <= 0.4 and abs(coin_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(coin_x - self.last_coin_x) < 1 and abs(coin_y - self.last_coin_y) < 1:
                    position_check = True

                self.coin_position.position.x = coin_x
                self.coin_position.position.y = coin_y

        else:
            while position_check:
                coin_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                coin_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                self.index = random.randrange(0, 13)
                print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.coin_position.position.x = coin_x_list[self.index]
                self.coin_position.position.y = coin_y_list[self.index]

        time.sleep(0.5)
        
        self.respawnModel()

        self.last_coin_x = self.coin_position.position.x
        self.last_coin_y = self.coin_position.position.y


        return self.coin_position.position.x, self.coin_position.position.y

    def getCoinDistace(self, robot_x, robot_y):
        coin_distance = round(math.hypot(self.coin_position.position.x - robot_x, self.coin_position.position.y - robot_y), 2)

        return coin_distance

    def coin_heading(self, yaw, bot_position_x, bot_position_y):
        coin_angle = math.atan2(self.coin_position.position.y - bot_position_y, self.coin_position.position.x - bot_position_x)

        heading = coin_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return heading