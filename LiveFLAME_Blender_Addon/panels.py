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
from bpy.types import Panel


# Draw Socket panel in Toolbar
class STREAMFLAME_PT_zmqConnector(Panel):
    """Interface to set and (dis)connect the ZeroMQ socket; Found in side panel of the 3D view
     (open by pressing `n` or dragging `<`)"""

    bl_label = "LiveFLAME"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "LiveFLAME"
    # bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        preferences = context.preferences.addons[__package__].preferences
        socket_settings = context.window_manager.socket_settings

        #   check if pyzmq is installed; will fail with an ImportError if not
        # if installed, will show interaction options: (dis)connect socket and whether to use dynamic object selection
        try:
            import zmq

            col = layout.column(align=True)
            row = col.row(align=True)
            col.label(text="Add FLAME Textured Head Figure:")
            col.prop(socket_settings, "flame_gender")
            col.operator("scene.flame_add_gender", text="Add to scene")
            col.separator()
            col.separator()

            # connection information
            col.label(text="PyZMQ Setup:")
            col.separator()
            row = col.row(align=True)
            row.prop(preferences, "socket_ip", text="ip")
            row.prop(preferences, "socket_port", text="port")
            row.prop(preferences, "socket_topic", text="topic")
            col.separator()
            col.separator()

            # whether if previous selection is remembered or always use current selected objects
            col.label(text="Remeber The Selected Object:")
            col.separator()
            row = col.row(align=True)
            col.prop(socket_settings, "dynamic_object")
            col.separator()
            # if our socket hasn't connected yet
            row = col.row(align=True)
            if not socket_settings.socket_connected:
                col.operator("socket.connect_subscriber")  # , text="Connect Socket"
            else:
                col.operator("socket.connect_subscriber", text="Disconnect Socket")
                col.prop(socket_settings, "msg_received")
            col.separator()
            col.separator()
          

            row = layout.row()
            col.label(text="Dynamic Options:")
            row.prop(socket_settings, 'facial_configuration')
            row.prop(socket_settings, 'rotate_head')
            row = layout.row()
            row.prop(socket_settings, 'keyframing')
            row.prop(socket_settings, 'mirror_head')
            row = layout.row()
            row.prop(socket_settings, 'flame_corrective_poseshapes')

            row = layout.row()
            col = layout.column(align=True)
            col.label(text="Fake Teeth Effect:")
            row = col.row(align=True)
            row.operator("object.flame_close_color_mesh", text="Close mesh")
            row.operator("object.flame_restore_mesh", text="Open mesh")
            col.separator()

            col.label(text="Reset Pose and Expressions:")
            row = col.row(align=True)
            split = row.split(factor=0.50, align=True)
            split.operator("object.flame_reset_pose", text="Reset Pose")
            split.operator("object.flame_reset_expressions", text="Reset Expression")

        # if not installed, show button that enables & updates pip, and pip installs pyzmq
        except ImportError:
            # keep track of how our installation is going
            install_props = context.window_manager.install_props

            # button: enable pip and install pyzmq if not available
            layout.operator("pipzmq.pip_pyzmq")
            # show status messages (kinda cramped)
            layout.prop(install_props, "install_status")


def register():
    bpy.utils.register_class(STREAMFLAME_PT_zmqConnector)


def unregister():
    bpy.utils.unregister_class(STREAMFLAME_PT_zmqConnector)


if __name__ == "__main__":
    register()
