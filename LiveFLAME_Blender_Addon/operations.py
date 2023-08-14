# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####



import bpy
import sys
import subprocess  
from pathlib import Path 
import json

import bmesh
from mathutils import Vector
from math import radians
import numpy as np
import os
import time

import math
import traceback

flame_hole_faces = [(1595,1743,1863),(1595,1596,1743),(1596,1747,1743),(1747,1748,1743),(1743,1740,1863),(1740,1666,1831),(1863,1740,1831),(1666,1667,1836),(1667,3515,1853),(1836,1667,1853),(3515,2784,2942),(2784,2783,2934),(2783,2855,2931),(2855,2858,2946),(2858,2863,2862),(2862,2732,2858),(2732,2731,2858),(2731,2709,2946),(2709,2710,2944),(2946,2709,2944),(2946,2931,2855),(2931,2934,2783),(2934,2942,2784),(2942,3498,3515),(3498,1853,3515),(1836,1831,1666),(1863,1861,1573),(1861,1574,1573),(1573,1595,1863),(2731,2946,2858),(3323,3372,3331),(3257,3256,3261),(3256,3258,3261),(3258,3259,3290),(3286,3380,3360),(3380,3358,3357),(3258,3290,3261),(3290,3286,3261),(3219,3257,3262),(3227,3220,3273),(3220,3219,3229),(3273,3230,3274),(3229,3230,3220),(3220,3230,3273),(3262,3229,3219),(3257,3261,3262),(3380,3357,3360),(3357,3355,3360),(3355,3356,3360),(3356,3322,3361),(3322,3323,3330),(3356,3361,3360),(3323,3328,3372),(3372,3373,3331),(3331,3330,3323),(3330,3361,3322),(3360,3249,3286),(3249,3261,3286)]
flame_default_faces = 9976
flame_pose_length = 15
flame_necessary_sk = ['Base','Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5', 'Exp6', 'Exp7', 'Exp8', 'Exp9', 'Exp10', 'Exp11', 'Exp12', 'Exp13', 'Exp14', 'Exp15', 'Exp16', 'Exp17', 'Exp18', 'Exp19', 'Exp20', 'Exp21', 'Exp22', 'Exp23', 'Exp24', 'Exp25', 'Exp26', 'Exp27', 'Exp28', 'Exp29', 'Exp30', 'Exp31', 'Exp32', 'Exp33', 'Exp34', 'Exp35', 'Exp36', 'Exp37', 'Exp38', 'Exp39', 'Exp40', 'Exp41', 'Exp42', 'Exp43', 'Exp44', 'Exp45', 'Exp46', 'Exp47', 'Exp48', 'Exp49', 'Exp50', 'Pose1', 'Pose2', 'Pose3', 'Pose4', 'Pose5', 'Pose6', 'Pose7', 'Pose8', 'Pose9', 'Pose10', 'Pose11', 'Pose12', 'Pose13', 'Pose14', 'Pose15', 'Pose16', 'Pose17', 'Pose18']
dummy_pose = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def create_material(mat_name, diffuse_color=(1,1,1,1)):
    mat = bpy.data.materials.new(name=mat_name)
    mat.diffuse_color = diffuse_color
    return mat

def rodrigues_from_pose(armature, bone_name):
    # Ensure that rotation mode is AXIS_ANGLE so the we get a correct readout of current pose
    armature.pose.bones[bone_name].rotation_mode = 'AXIS_ANGLE'
    axis_angle = armature.pose.bones[bone_name].rotation_axis_angle

    angle = axis_angle[0]

    rodrigues = Vector((axis_angle[1], axis_angle[2], axis_angle[3]))
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues


class SOCKET_OT_connect_subscriber(bpy.types.Operator):
    """Manages the binding of a subscriber ZeroMQ socket. Select FLAME mesh in Object mode to"""
    # Use this as a tooltip for menu items and buttons.

    bl_idname = "socket.connect_subscriber"  
    bl_label = "Connect socket" 
    bl_options = {'REGISTER', 'UNDO'} 
    statetest = "Nothing yet..."

    @classmethod
    def poll(cls, context):
        try:
          
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context): 
        """Either sets-up a ZeroMQ subscriber socket and make timed_msg_poller active,
        or turns-off the timed function and shuts-down the socket."""

        # install pyzmq for the belnder
        import zmq
        
        preferences = context.preferences.addons[__package__].preferences
        # get access to our Properties in STREAMFLAMEProperties() (properties.py)
        self.socket_settings = context.window_manager.socket_settings

        # connect our socket if it wasn't and call Blender's timer function on self.timed_msg_poller
        if not self.socket_settings.socket_connected:
            self.frame_start = context.scene.frame_current

            #FPS counter
            self.msg_count = 0
            self.fps_count = 0
            self.start_time = time.time()
            print(f"start time: {self.start_time}")
            self.dsp_r_at_x = 1  # displays the frame rate every 1 second


            self.report({'INFO'}, "Connecting ZeroMQ socket...")
            # create a ZeroMQ context
            self.zmq_ctx = zmq.Context().instance()
            # connect to ip and port specified in interface (blendzmq_panel.py)
            self.url = f"tcp://{preferences.socket_ip}:{preferences.socket_port}"

            # store our connection in Blender's WindowManager for access in self.timed_msg_poller()
            bpy.types.WindowManager.socket_sub = self.zmq_ctx.socket(zmq.SUB)
            bpy.types.WindowManager.socket_sub.connect(self.url)  # publisher connects to this (subscriber)
            bpy.types.WindowManager.socket_sub.setsockopt(zmq.SUBSCRIBE, f'{preferences.socket_topic}'.encode('ascii'))
            self.report({'INFO'}, "Sub connected to: {}\nWaiting for data...".format(self.url))

            # poller socket for checking server replies (synchronous - not sure how to use async with Blender)
            self.poller = zmq.Poller()
            self.poller.register(bpy.types.WindowManager.socket_sub, zmq.POLLIN)

            # let Blender know our socket is connected
            self.socket_settings.socket_connected = True

            # reference to selected objects at start of data stream;
            # a copy is made, because this is a pointer (which is updated when another object is selected)
            self.selected_obj = bpy.context.scene.view_layers[0].objects.active
            print("selected object",self.selected_obj)
            bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'INFO'}, f"selected object {self.selected_obj}")
            
            # delete unwanted shapekeys to make this tool faster to process
            self.shape_keys_local = self.selected_obj.data.shape_keys.key_blocks 
        
            for sk in self.selected_obj.data.shape_keys.key_blocks:
                if sk.name not in flame_necessary_sk:
                    self.selected_obj.shape_key_remove(key=sk)
                    
            self.report({'INFO'}, f"Delete shape and pose blendshapes from {self.selected_obj}. Now it has only {len(self.selected_obj.data.shape_keys.key_blocks)} shapekeys ")
        
            # have Blender call our data listening function in the background
            bpy.app.timers.register(self.timed_msg_poller)
        

        # stop ZMQ poller timer and disconnect ZMQ socket
        else:
            print(self.statetest)
            # cancel timer function with poller if active
            if bpy.app.timers.is_registered(self.timed_msg_poller):
                bpy.app.timers.unregister(self.timed_msg_poller())

            # Blender's property socket_connected might say connected, but it might actually be not;
            # e.g. on Add-on reload
            try:
                # close connection
                bpy.types.WindowManager.socket_sub.close()
                self.report({'INFO'}, "Subscriber socket closed")
            except AttributeError:
                self.report({'INFO'}, "Subscriber was socket not active")

            # let Blender know our socket is disconnected
            bpy.types.WindowManager.socket_sub = None
            self.socket_settings.socket_connected = False

            #reset message field
            self.socket_settings.msg_received = "Awaiting msg..."

        return {'FINISHED'}  # Lets Blender know the operator finished successfully.

    def timed_msg_poller(self):  # context
        """Keeps listening to flame values and uses that to move (previously) selected objects"""

        socket_sub = bpy.types.WindowManager.socket_sub
      

        # only keep running if socket reference exist (not None)
        if socket_sub:
            # get sockets with messages (0: don't wait for msgs)
            sockets = dict(self.poller.poll(0))
            # check if our sub socket has a message
            if socket_sub in sockets:
                # get the message
                capture_start = time.time()
                topic, timestamp, msg = socket_sub.recv_multipart()
                print(f"capture_latency: {(time.time() - capture_start)}")
                
                process_start = time.time()
                msg = msg.decode('utf-8')
                
                if self.msg_count <=1:
                    self.socket_settings.msg_received = "Streaming data..."
           
                
                if msg:
                    # turn json string to dict
                    msg = json.loads(msg)
                    ############### setting shape keys and bone roations #####
                    if self.socket_settings.dynamic_object:
                        # only active object (no need for a copy)
                        # if you need multiple object activated look into https://github.com/NumesSanguis/FACSvatar-Blender
                        self.selected_obj = bpy.context.scene.view_layers[0].objects.active

                        """
                        incoming json keys: confidence, timestamp, frame, smooth, flame, timestamp_utc 
                        """
                        try:
            
                            if 'frame' in msg:
                                insert_frame = self.frame_start + msg['frame']
                            else:
                                insert_frame = self.frame_start + self.msg_count

                            if msg['flame']:
                                
                                    # combine both update expression and update pose
                                    #print(insert_frame)
                                    shape_data_ = np.array(msg['flame'][:50])
                                    pose_data_ = np.array(msg['flame'][50:])
                                    #print(shaps_data_)
                            else:
                                 # keep running and check every 0.1 millisecond for new ZeroMQ messages
                                print("Incoming message containes no flame information")
                                self.report({'WARNING'}, "Incoming message containes no flame information")
                                return 0.001

                            # set blendshapes only if blendshape data is available and not empty
                            if self.socket_settings.facial_configuration and self.socket_settings.rotate_head:
                                self.set_frame_pose_expression(obj=self.selected_obj,shape_msg=shape_data_, pose_data=pose_data_,kf=insert_frame)
                            elif self.socket_settings.facial_configuration:
                                self.update_expression(obj=self.selected_obj,shape_msg= shape_data_ ,kf= insert_frame)
                            elif self.socket_settings.rotate_head:
                                self.update_pose(obj=self.selected_obj,pose_data= pose_data_ ,kf= insert_frame)
                            else:
                                print("tick an option")
                                #self.report({'INFO'}, "No pose data found in received msg")
                            
                        except:
                            traceback.print_exc()
                            self.report({'WARNING'}, "Object likely not a support model")


                    print(f"process_latency: {(time.time() - process_start)}")
                    ############# FPS counter ##########
                    self.msg_count += 1
                    self.fps_count += 1
                    # Just reading message FPS reached: 30 (incoming)
                    if (time.time() - self.start_time) > self.dsp_r_at_x:
                        #print("FPS: ", self.fps_count / (time.time() - self.start_time))
                        self.socket_settings.msg_received = f"FPS: {round(self.fps_count / (time.time() - self.start_time))}"
                        self.fps_count = 0
                        self.start_time = time.time()
            
                else:
                    self.socket_settings.msg_received = "Last message received."

            # keep running and check every 0.1 millisecond for new ZeroMQ messages
            return 0.001

    def update_expression(self, obj,shape_msg, kf):
        """ change the method due to slowness of setting all these shape keys"""
        
        for j in range(1,51):
            obj.data.shape_keys.key_blocks[f"Exp{j}"].value = shape_msg[j-1]

        bpy.ops.object.flame_update_joint_locations('EXEC_DEFAULT')

    def update_pose(self, obj, pose_data, kf):
        if (obj.type == 'ARMATURE'):
            armature = obj
        else:
            armature = obj.parent
     
        # Change rotation mode from AXIS_ANGLE to XYZ to see changes
        # update the bone positions
        """
        """
        if len(pose_data):
            global_quat = pose_data[:4]
            jaw_quat = pose_data[4:]
            print(global_quat, jaw_quat)
             
            #print(f"x:{x},y:{y},z:{z} ")
            try:
                armature.pose.bones["neck"].rotation_mode = 'QUATERNION'
                armature.pose.bones["neck"].rotation_quaternion = global_quat


                armature.pose.bones["jaw"].rotation_mode = 'QUATERNION'
                armature.pose.bones["jaw"].rotation_quaternion = jaw_quat
            except:
                traceback.print_exc()
                self.report({'WARNING'}, "Something wrong with neck and jaw data")
        else:
            self.report({'WARNING'}, "No Neck Pose available")

        if self.socket_settings.flame_corrective_poseshapes:
            # Update corrective poseshapes
            bpy.ops.object.flame_set_poseshapes('EXEC_DEFAULT')

         # find an efficient way to implement as it could exponentially grow in computation
        if self.socket_settings.keyframing:
            pass
            #armature.pose.bones["neck"].keyframe_insert(data_path="rotation_euler", frame=kf)
            #armature.pose.bones["jaw"].keyframe_insert(data_path="rotation_euler", frame=kf)

    def set_frame_pose_expression(self, obj, shape_msg, pose_data, kf):
        if (obj.type == 'ARMATURE'):
            armature = obj
        else:
            armature = obj.parent

        for j in range(1,51):
            obj.data.shape_keys.key_blocks[f"Exp{j}"].value = shape_msg[j-1]

     
        # update the bone positions
        """
        """
        if len(pose_data):
            global_quat = pose_data[:4]
            jaw_quat = pose_data[4:]
             
            #print(f"x:{x},y:{y},z:{z} ")
            try:
                armature.pose.bones["neck"].rotation_mode = 'QUATERNION'
                armature.pose.bones["neck"].rotation_quaternion = global_quat

        
                armature.pose.bones["jaw"].rotation_mode = 'QUATERNION'
                armature.pose.bones["jaw"].rotation_quaternion = jaw_quat
            except:
                traceback.print_exc()
                self.report({'WARNING'}, "Something wrong with neck and jaw data")
        else:
            self.report({'WARNING'}, "No Neck Pose available")


        bpy.ops.object.flame_update_joint_locations('EXEC_DEFAULT')

        if self.socket_settings.flame_corrective_poseshapes:
            # Update corrective poseshapes
            bpy.ops.object.flame_set_poseshapes('EXEC_DEFAULT')

         # save as key frames if enabled
        if self.socket_settings.keyframing:
            armature.pose.bones["neck"].keyframe_insert(data_path="rotation_euler", frame=kf)
            armature.pose.bones["jaw"].keyframe_insert(data_path="rotation_euler", frame=kf)


class PIPZMQ_OT_pip_pyzmq(bpy.types.Operator):
    """Enables and updates pip, and installs pyzmq"""  # Use this as a tooltip for menu items and buttons.

    bl_idname = "pipzmq.pip_pyzmq"  # Unique identifier for buttons and menu items to reference.
    bl_label = "Enable pip & install pyzmq"  # Display name in the interface.
    bl_options = {'REGISTER'}

    def execute(self, context):  # execute() is called when running the operator.
        install_props = context.window_manager.install_props

        # pip in Blender:
        # https://blender.stackexchange.com/questions/139718/install-pip-and-packages-from-within-blender-os-independently/
        # pip 2.81 issues: https://developer.blender.org/T71856

        # no pip enabled by default version < 2.81
        install_props.install_status = "Preparing to enable pip..."
        self.report({'INFO'}, "Preparing to enable pip...")
        if bpy.app.version[0] == 2 and bpy.app.version[1] < 81:
            # find python binary OS independent (Windows: bin\python.exe; Linux: bin/python3.7m)
            py_path = Path(sys.prefix) / "bin"
            py_exec = str(next(py_path.glob("python*")))  # first file that starts with "python" in "bin" dir

            if subprocess.call([py_exec, "-m", "ensurepip"]) != 0:
                install_props.install_status += "\nCouldn't activate pip."
                self.report({'ERROR'}, "Couldn't activate pip.")
                return {'CANCELLED'}

        # from 2.81 pip is enabled by default
        else:
            try:
                # will likely fail the first time, but works after `ensurepip.bootstrap()` has been called once
                import pip
            except ModuleNotFoundError as e:
                # only first attempt will reach here
                print("Pip import failed with: ", e)
                install_props.install_status += "\nPip not activated, trying bootstrap()"
                self.report({'ERROR'}, "Pip not activated, trying bootstrap()")
                try:
                    import ensurepip
                    ensurepip.bootstrap()
                except:  # catch *all* exceptions
                    e = sys.exc_info()[0]
                    install_props.install_status += "\nPip not activated, trying bootstrap()"
                    self.report({'ERROR'}, "Pip not activated, trying bootstrap()")
                    print("bootstrap failed with: ", e)
            py_exec = bpy.app.binary_path_python

        # TODO check permission rights
        # TODO Windows ask for permission:
        #  https://stackoverflow.com/questions/130763/request-uac-elevation-from-within-a-python-script

        install_props.install_status += "\nPip activated! Updating pip..."
        self.report({'INFO'}, "Pip activated! Updating pip...")

        # pip update
        try:
            print("Trying pip upgrade")
            output = subprocess.check_output([py_exec, '-m', 'pip', 'install', '--upgrade', 'pip'])
            print(output)
        except subprocess.CalledProcessError as e:
            install_props.install_status += "\nCouldn't update pip. Please restart Blender and try again."
            self.report({'ERROR'}, "Couldn't update pip. Please restart Blender and try again.")
            print(e.output)
            return {'CANCELLED'}
        install_props.install_status += "\nPip working! Installing pyzmq..."
        self.report({'INFO'}, "Pip working! Installing pyzmq...")

        # pyzmq pip install
        try:
            print("Trying pyzmq install")
            output = subprocess.check_output([py_exec, '-m', 'pip', 'install', 'pyzmq'])
            print(output)
        except subprocess.CalledProcessError as e:
            install_props.install_status += "\nCouldn't install pyzmq."
            self.report({'ERROR'}, "Couldn't install pyzmq.")
            print(e.output)
            return {'CANCELLED'}

        install_props.install_status += "\npyzmq installed! READY!"
        self.report({'INFO'}, "pyzmq installed! READY!")

        return {'FINISHED'}  # Lets Blender know the operator finished successfully


####################### FLAME Operations ###################
class FlameAddGender(bpy.types.Operator):
    bl_idname = "scene.flame_add_gender"
    bl_label = "Add"
    bl_description = ("Add FLAME model of selected gender to scene")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        gender = context.window_manager.socket_settings.flame_gender
        print("Adding gender: " + gender)

        path = os.path.dirname(os.path.realpath(__file__))
        objects_path = os.path.join(path, "data", "flame2020_textured_%s.blend" % (gender), "Object")
        object_name = "FLAME-" + gender

        bpy.ops.wm.append(filename=object_name, directory=str(objects_path))

        # Select imported FACE mesh
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.objects[object_name].select_set(True)

        return {'FINISHED'}
    

# this is called after every function
class FlameUpdateJointLocations(bpy.types.Operator):
    bl_idname = "object.flame_update_joint_locations"
    bl_label = "Update joint locations"
    bl_description = ("Update joint locations after shape/expression changes")
    bl_options = {'REGISTER', 'UNDO'}

    j_regressor = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        if self.j_regressor is None:
            path = os.path.dirname(os.path.realpath(__file__))
            regressor_path = os.path.join(path, "data", "flame2020_joint_regressor.npz")
            with np.load(regressor_path) as data:
                self.j_regressor = data['joint_regressor']


        # Store current bone rotations
        armature = obj.parent

        bone_rotations = {}
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = 'AXIS_ANGLE'
            axis_angle = pose_bone.rotation_axis_angle
            bone_rotations[pose_bone.name] = (axis_angle[0], axis_angle[1], axis_angle[2], axis_angle[3])

        # Set model in default pose
        for bone in armature.pose.bones:
            bpy.ops.object.flame_reset_poseshapes('EXEC_DEFAULT')
            bone.rotation_mode = 'AXIS_ANGLE'
            bone.rotation_axis_angle = (0, 0, 1, 0)

        # Reset corrective poseshapes if used
        if context.window_manager.socket_settings.flame_corrective_poseshapes:
            bpy.ops.object.flame_reset_poseshapes('EXEC_DEFAULT')

        # Get vertices with applied skin modifier
        depsgraph = context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_from_eval = object_eval.to_mesh()

        # Get Blender vertices as numpy matrix
        vertices_np = np.zeros((len(mesh_from_eval.vertices)*3), dtype=np.float)
        mesh_from_eval.vertices.foreach_get("co", vertices_np)
        vertices_matrix = np.reshape(vertices_np, (len(mesh_from_eval.vertices), 3))
        object_eval.to_mesh_clear() # Remove temporary mesh

        joint_locations = self.j_regressor @ vertices_matrix

        # Set new bone joint locations
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        for index, bone in enumerate(armature.data.edit_bones):

            if index == 0:
                continue # ignore root bone
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.01)

            bone_start = Vector(joint_locations[index])
            bone.translate(bone_start)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj

        # Restore pose
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = 'AXIS_ANGLE'
            pose_bone.rotation_axis_angle = bone_rotations[pose_bone.name]

        # Restore corrective poseshapes if used
        if context.window_manager.socket_settings.flame_corrective_poseshapes:
            bpy.ops.object.flame_set_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}

#poseshape
class FlameSetPoseshapes(bpy.types.Operator):

    bl_idname = "object.flame_set_poseshapes"
    bl_label = "Set poseshapes"
    bl_description = ("Sets corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    def rodrigues_to_mat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]])
        return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Calculate weights of pose corrective blend shapes
    def rodrigues_to_posecorrective_weight(self, pose):
        rod_rots = np.asarray(pose).reshape(5, 3)
        mat_rots = [self.rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return(bshapes)

    def execute(self, context):
        obj = bpy.context.object

        # Get armature pose in rodrigues representation
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        neck = rodrigues_from_pose(armature, "neck")
        jaw = rodrigues_from_pose(armature, "jaw")

        pose = [0.0] * flame_pose_length
        pose[3] = neck[0]
        pose[4] = neck[1]
        pose[5] = neck[2]

        pose[6] = jaw[0]
        pose[7] = jaw[1]
        pose[8] = jaw[2]

        # print("Current pose: " + str(pose))

        poseweights = self.rodrigues_to_posecorrective_weight(pose)

        # Set weights for pose corrective shape keys
        for index, weight in enumerate(poseweights):
            if index >= 18:
                break
            obj.data.shape_keys.key_blocks["Pose%d" % (index+1)].value = weight

        # Set checkbox without triggering update function
        context.window_manager.socket_settings["flame_corrective_poseshapes"] = True

        return {'FINISHED'}

class FlameResetPoseshapes(bpy.types.Operator):
    bl_idname = "object.flame_reset_poseshapes"
    bl_label = "Reset poseshapes"
    bl_description = ("Resets corrective poseshapes for current pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'ARMATURE':
            obj = bpy.context.object.children[0]

#        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Pose"):
                key_block.value = 0.0

        return {'FINISHED'}


#pose
class FlameResetPose(bpy.types.Operator):
    bl_idname = "object.flame_reset_pose"
    bl_label = "Reset pose"
    bl_description = ("Resets pose to default zero pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        armature = obj.parent

        for bone in armature.pose.bones:
            bone.rotation_mode = 'AXIS_ANGLE'
            bone.rotation_axis_angle = (0, 0, 1, 0)

        # Reset sliders without updating pose
        context.window_manager.flame_tool["flame_neck_yaw"] = 0.0
        context.window_manager.flame_tool["flame_neck_pitch"] = 0.0
        context.window_manager.flame_tool["flame_jaw"] = 0.0

        # Reset corrective pose shapes
        bpy.ops.object.flame_reset_poseshapes('EXEC_DEFAULT')

        return {'FINISHED'}


#expression 
class FlameResetExpressions(bpy.types.Operator):
    bl_idname = "object.flame_reset_expressions"
    bl_label = "Reset expressions"
    bl_description = ("Resets all blend shape keys for expressions")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Exp"):
                key_block.value = 0.0

        bpy.ops.object.flame_update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}


# teeth and bottom 
class FlameCloseColorMesh(bpy.types.Operator):
    bl_idname = "object.flame_close_color_mesh"
    bl_label = "Close mesh"
    bl_description = ("Closes all open holes in the FLAME mesh with a colored mesh")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        # Get a bmesh representation
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        if len(bm.faces) > flame_default_faces:
            print("FLAME: Holes are already filled")
        else:
            bm.verts.ensure_lookup_table()

            for (i0, i1, i2) in flame_hole_faces:
                bm.faces.new((bm.verts[i0-1], bm.verts[i1-1], bm.verts[i2-1]))

        # Write the modified bmesh back to the mesh
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)

        # add color to the newly created faces
        color_name = 'White'
        color_value = (1,1,1,1)
        material = create_material(color_name, color_value)

        # Append both Materials to the created object
        if color_name not in obj.data.materials:
            obj.data.materials.append(material)
        else: 
            print("color_name exist")

        faces_color = bm.faces[flame_default_faces:]
        print(f"setting material {obj.data.materials.find(color_name)} index for faces")
        for faces_ in faces_color:
            faces_.material_index = obj.data.materials.find(color_name)

        bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}

class FlameRestoreMesh(bpy.types.Operator):
    bl_idname = "object.flame_restore_mesh"
    bl_label = "Restore mesh"
    bl_description = ("Restores all open holes in the FLAME mesh")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        # Get a bmesh representation
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)

        if len(bm.faces) == flame_default_faces:
            print("FLAME: Mesh already restored to default state")
        else:
            bm.faces.ensure_lookup_table()

            faces_delete = bm.faces[flame_default_faces:]
            bmesh.ops.delete(bm, geom=faces_delete, context='FACES') # EDGES_FACES

        # Write the modified bmesh back to the mesh
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)

        bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}




classes = (FlameResetExpressions,
    FlameUpdateJointLocations,
    FlameSetPoseshapes,
    FlameResetPoseshapes,
    FlameResetPose,
    PIPZMQ_OT_pip_pyzmq,
    SOCKET_OT_connect_subscriber,
    FlameCloseColorMesh,
    FlameRestoreMesh,
    FlameAddGender
    )
    
def register():
   
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    #bpy.utils.register_class(PIPZMQ_OT_pip_pyzmq)
    #bpy.utils.register_class(SOCKET_OT_connect_subscriber)
    


def unregister():
    #bpy.utils.unregister_class(SOCKET_OT_connect_subscriber)
    #bpy.utils.register_class(PIPZMQ_OT_pip_pyzmq)
   
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

if __name__ == "__main__":
    register()