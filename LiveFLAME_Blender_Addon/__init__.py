# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "LiveFLAME",
    "author" : "",
    "description" : "Live Expressions and HeadMovements",
    "blender" : (2, 80, 0),
    "version" : (2022, 8, 15),
    "location": "View3D",
    "warning" : "",
    "description": "LiveFLAME stream FLAME via socket",
    "category" : "Object"
}

# add-on is being reloaded
if "bpy" in locals():
    print("reloading .py files")
    import importlib

    from . import properties
    importlib.reload(properties)
    from . import panels
    importlib.reload(panels)
    from . import operations
    importlib.reload(operations)
# first time loading add-on
else:
    print("importing .py files")
    import bpy
    from . import properties
    from . import panels
    from . import operations

from bpy.types import AddonPreferences
from bpy.props import (
    PointerProperty,
    StringProperty,
)
from . properties import PIPSTREAMFLAMEProperties, STREAMFLAMEProperties
from . panels import STREAMFLAME_PT_zmqConnector
from . operations import SOCKET_OT_connect_subscriber, PIPZMQ_OT_pip_pyzmq 
from . operations import FlameResetPose, FlameResetPoseshapes,FlameSetPoseshapes, FlameResetExpressions, FlameUpdateJointLocations, FlameCloseColorMesh, FlameRestoreMesh, FlameAddGender


# Add-on Preferences
class STREAMFLAMEPreferences(AddonPreferences):
    """Remember ip and port number as addon preference (across Blender sessions)

    Editable in UI interface, or `Edit -> Preferences... -> Add-ons -> Development: FACSvatar -> Preferences`"""

    bl_idname = __name__


    socket_ip: StringProperty(name="Socket ip",
                              description="IP of ZMQ publisher socket",
                              default="127.0.0.1",
                              )
    socket_port: StringProperty(name="Socket port",
                                description="Port of ZMQ publisher socket",
                                default="5572",
                                )
    socket_topic: StringProperty(name="Socket Topic",
                                description="Topic to filter messages",
                                default="flame",
                                )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Socket connection settings:")

        row = layout.row(align=True)
        row.prop(self, "socket_ip", text="ip")
        row.prop(self, "socket_port", text="port")
        row.prop(self, "socket_topic", text="topic")


# Define Classes to register
classes = (
    PIPSTREAMFLAMEProperties,
    STREAMFLAMEProperties,
    PIPZMQ_OT_pip_pyzmq,
    SOCKET_OT_connect_subscriber,
    STREAMFLAMEPreferences,
    STREAMFLAME_PT_zmqConnector,
    FlameResetExpressions,
    FlameUpdateJointLocations,
    FlameSetPoseshapes,
    FlameResetPoseshapes,
    FlameResetPose,
    FlameCloseColorMesh,
    FlameRestoreMesh,
    FlameAddGender

    
    )


# one-liner to (un)register if no property registration was needed
# register, unregister = bpy.utils.register_classes_factory(classes)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.install_props = PointerProperty(type=PIPSTREAMFLAMEProperties)
    bpy.types.WindowManager.socket_settings = PointerProperty(type=STREAMFLAMEProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.WindowManager.socket_settings
    del bpy.types.WindowManager.install_props


if __name__ == "__main__":
    register()
