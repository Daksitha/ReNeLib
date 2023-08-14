from pathlib import Path
from tqdm import tqdm
import json
import os
import numpy as np
import pandas as pd
import torch
from .rotation_conversion import angle_axis_to_quaternion, quaternion_to_euler_xyz, batch_axis2euler
import time
###### transformation matrix
tr_matrix_pth = Path(os.path.dirname(os.path.abspath(__file__))) /"data" / "facial_expression_matrix.npy"
if tr_matrix_pth.exists():
    transform_matrx = np.load(tr_matrix_pth)
else:
    raise RuntimeError(f"transform_matrx is not {tr_matrix_pth}, please create")
"""
names of 96 shapekey in charamel character
"""
shape_key_names = ['brows_up', 'brows_down', 'brows_tight_up', 'brows_tight_down', 'lids_upper_rotated_closing', 'lids_wide_open', 'lids_lower_up',
'lids_lower_down', 'lids_upper_up', 'lids_upper_down', 'nose_up', 'nose_down', 'nosewings_wide', 'nosewings_tight', 'cheeks_up',
'cheeks_down', 'mouth_up', 'mouth_down', 'mouth_open', 'mouth_tight', 'mouth_wide', 'mouth_rotate_in', 'mouth_rotate_out',
'jaw_front', 'jaw_back', 'jaw_down', 'tongue_tip_gum_up', 'tongue_tip_gum_down', 'tongue_tip_teeth_upper', 'tongue_tip_teeth_lower',
'tongue_tip_stick_out', 'mimic_blink', 'mimic_kiss', 'mimic_winkl', 'mimic_lidl_close', 'mimic_lidr_close', 'mimic_brows_down',
'mimic_brows_down_right', 'mimic_brows_down_left', 'mimic_brows_up', 'mimic_brows_up_right', 'mimic_brows_up_left', 'mimic_blow',
'mimic_breath', 'browouterupleft', 'browouterupright', 'browinnerup', 'browdownleft', 'browdownright', 'eyeblinkleft', 'eyeblinkright',
'eyelookinleft', 'eyelookoutleft', 'eyelookupleft', 'eyelookdownleft', 'eyelookinright', 'eyelookoutright', 'eyelookupright', 'eyelookdownright',
'eyesquintleft', 'eyesquintright', 'eyewideleft', 'eyewideright', 'cheekpuff', 'cheeksquintleft', 'cheeksquintright',
'nosesneerleft', 'nosesneerright', 'jawforward', 'jawopen', 'jawleft', 'jawright', 'mouthclose', 'mouthfunnel',
'mouthdimpleleft', 'mouthdimpleright', 'mouthfrownleft', 'mouthfrownright', 'mouthsmileleft', 'mouthsmileright', 'mouthpucker',
                   'mouthlowerdownleft', 'mouthlowerdownright', 'mouthleft', 'mouthright', 'tongueout', 'mouthpressleft', 'mouthpressright',
                   'mouthrollupper', 'mouthrolllower', 'mouthstretchleft', 'mouthstretchright', 'mouthshruglower', 'mouthshrugupper',
                   'mouthupperupleft', 'mouthupperupright']
pose_names = ['globx','globy', 'globz', 'jawx', 'jawy', 'jawz']
Flame_arr = [[0.4683678150177002, 0.565824568271637, 0.10815038532018661, -0.005692719016224146, -0.3385692536830902,
             -0.5257269144058228,
             0.36900293827056885, 0.17903219163417816, 0.05094520002603531, -0.24144010245800018, 0.1777740865945816,
             -0.15688754618167877,
             -0.24329769611358643, -0.3064752519130707, 0.5565656423568726, -0.07659304887056351, -0.22986115515232086,
             -0.1410302072763443,
             -0.0300979632884264, 0.19591045379638672, 0.192985400557518, -0.1196194440126419, 0.0012663668021559715,
             0.0686517059803009,
             0.12743133306503296, -0.024382170289754868, -0.1628689467906952, 0.06095920503139496, 0.09950602054595947,
             -0.05893649905920029,
             -0.026644213125109673, -0.1025707945227623, 0.08184178918600082, -0.005937675014138222,
             0.07615204155445099, -0.11418697983026505,
             0.06334573775529861, -0.038835559040308, 0.03376869112253189, -0.11200832575559616, 0.037162043154239655,
             -0.07455194741487503,
             0.1503240317106247, 0.024716122075915337, -0.03262915089726448, -0.007129407487809658,
             -0.18705272674560547, 0.17004796862602234,
             -0.008296679705381393, -0.05324096232652664, 0.15935854613780975, -0.01499017234891653,
             0.04201154410839081, 0.054376035928726196,
             0.02477172203361988, 0.007921779528260231]]

multi_array = [[0.4683678150177002, 0.565824568271637, 0.10815038532018661, -0.005692719016224146, -0.3385692536830902,
             -0.5257269144058228,
             0.36900293827056885, 0.17903219163417816, 0.05094520002603531, -0.24144010245800018, 0.1777740865945816,
             -0.15688754618167877,
             -0.24329769611358643, -0.3064752519130707, 0.5565656423568726, -0.07659304887056351, -0.22986115515232086,
             -0.1410302072763443,
             -0.0300979632884264, 0.19591045379638672, 0.192985400557518, -0.1196194440126419, 0.0012663668021559715,
             0.0686517059803009,
             0.12743133306503296, -0.024382170289754868, -0.1628689467906952, 0.06095920503139496, 0.09950602054595947,
             -0.05893649905920029,
             -0.026644213125109673, -0.1025707945227623, 0.08184178918600082, -0.005937675014138222,
             0.07615204155445099, -0.11418697983026505,
             0.06334573775529861, -0.038835559040308, 0.03376869112253189, -0.11200832575559616, 0.037162043154239655,
             -0.07455194741487503,
             0.1503240317106247, 0.024716122075915337, -0.03262915089726448, -0.007129407487809658,
             -0.18705272674560547, 0.17004796862602234,
             -0.008296679705381393, -0.05324096232652664, 0.15935854613780975, -0.01499017234891653,
             0.04201154410839081, 0.054376035928726196,
             0.02477172203361988, 0.007921779528260231],[0.4683678150177002, 0.565824568271637, 0.10815038532018661, -0.005692719016224146, -0.3385692536830902,
             -0.5257269144058228,
             0.36900293827056885, 0.17903219163417816, 0.05094520002603531, -0.24144010245800018, 0.1777740865945816,
             -0.15688754618167877,
             -0.24329769611358643, -0.3064752519130707, 0.5565656423568726, -0.07659304887056351, -0.22986115515232086,
             -0.1410302072763443,
             -0.0300979632884264, 0.19591045379638672, 0.192985400557518, -0.1196194440126419, 0.0012663668021559715,
             0.0686517059803009,
             0.12743133306503296, -0.024382170289754868, -0.1628689467906952, 0.06095920503139496, 0.09950602054595947,
             -0.05893649905920029,
             -0.026644213125109673, -0.1025707945227623, 0.08184178918600082, -0.005937675014138222,
             0.07615204155445099, -0.11418697983026505,
             0.06334573775529861, -0.038835559040308, 0.03376869112253189, -0.11200832575559616, 0.037162043154239655,
             -0.07455194741487503,
             0.1503240317106247, 0.024716122075915337, -0.03262915089726448, -0.007129407487809658,
             -0.18705272674560547, 0.17004796862602234,
             -0.008296679705381393, -0.05324096232652664, 0.15935854613780975, -0.01499017234891653,
             0.04201154410839081, 0.054376035928726196,
             0.02477172203361988, 0.007921779528260231], np.linspace(3, 4, num=56)]


def create_matrix_from_json(parent_path):
    file_list = os.listdir(parent_path)
    min_arr = []
    max_arr = []
    # json_keys = None
    for filename in sorted(file_list, key=first_two_digit_sort):

        file = Path(os.path.dirname(os.path.abspath(__file__))) /"data" / "facial_expression"/filename
        with open(str(file), 'rb') as thefile:
            j_data = json.load(thefile)

            tmp_arr = []
            tmp_key = []

            if "min" in filename:
                for k in j_data:
                    tmp_arr.append(j_data[k])

                    # if json_keys == None:
                    #     tmp_key.append(k)
                min_arr.append(tmp_arr)
                # if json_keys == None:
                #     json_keys= tmp_key

            elif "max" in filename:
                for k in j_data:
                    tmp_arr.append(j_data[k])
                max_arr.append(tmp_arr)

            else:
                raise RuntimeError("Invalid file name")

    # Do stuff to each file
    max_nparr = np.array(max_arr, dtype=np.float32)
    min_nparr = np.array(min_arr, dtype=np.float32)
    concat_nparr = np.concatenate((max_nparr, min_nparr), axis=0)

    print(f"shapes: {max_nparr.shape},{min_nparr.shape}, Concatanated array: {concat_nparr.shape}")
    matrix_file = Path(os.path.dirname(os.path.abspath(__file__))) /"data" / "facial_expression_matrix.npy"
    m_csv_file = Path(os.path.dirname(os.path.abspath(__file__))) /"data" / "facial_expression_matrix.csv"

    print(f"save this matrix to {matrix_file} ")
    np.save(matrix_file,concat_nparr )
    # extra save for visibility
    np.savetxt(m_csv_file, concat_nparr, delimiter=",")


# grab last 4 characters of the file name:
def first_two_digit_sort(x):
    return (int(x.split("_")[0]))


def show_indices(obj, indices):
    for k, v in obj.items() if isinstance(obj, dict) else enumerate(obj):
        if isinstance(v, (dict, list)):
            yield from show_indices(v, indices + [k])
        else:
            yield indices + [k], v

# def angle_axis_quat_to_eurler(flame_pose):
#     # convert axis angles to euler
#     # first 3 axis angles are global and the last three are jaw pose
#     print("flame_pose shape:", flame_pose.shape)
#     posecode_tensor = torch.from_numpy(flame_pose)
#     try:
#
#         glob_euler_xyz = batch_axis2euler(posecode_tensor[:,:3])  # Nx4
#         jaw_euler_xyz = batch_axis2euler(posecode_tensor[:,3:])  # Nx4
#         print("glob_euler_xyz",glob_euler_xyz.shape,"glob_euler_xyz",glob_euler_xyz.shape)
#     except IndexError as ie:
#         print(ie)
#
#     #glob_euler_xyz = quaternion_to_euler_xyz(glob_qat).numpy()
#     #jaw_euler_xyz = quaternion_to_euler_xyz(jaw_qat).numpy()
#
#     return glob_euler_xyz, jaw_euler_xyz


def angle_axis_to_eurler_bactch(flame_pose):
    # convert axis angles to euler
    # first 3 axis angles are global and the last three are jaw pose
    print("flame_pose shape:", flame_pose.shape)
    posecode_tensor = torch.from_numpy(flame_pose)

    glob_qat = angle_axis_to_quaternion(posecode_tensor[:, :3])  # Nx4
    jaw_qat = angle_axis_to_quaternion(posecode_tensor[:, 3:])  # Nx4
    print(glob_qat.shape, jaw_qat.shape)

    glob_euler_xyz = quaternion_to_euler_xyz(glob_qat).numpy()
    jaw_euler_xyz = quaternion_to_euler_xyz(jaw_qat).numpy()

    return glob_euler_xyz, jaw_euler_xyz


def flame_transform_vector_to_ar(flame_fdim, dim=20):

    rpt_flame_fdim = np.concatenate((flame_fdim, flame_fdim), axis=1)

    # normalise the data by 3 as flame shapekey value range from [-3,0][0,3]
    flame_transform_vec = rpt_flame_fdim / 3
    #print(flame_transform_vec)

    # clip fist 20 (default) value: max value [0,1] and last 20 values [-1,0]
    flame_transform_vec[:,0:dim] = np.clip(flame_transform_vec[:,0:dim], a_min=0, a_max=1)
    flame_transform_vec[:,dim:-1] = np.clip(flame_transform_vec[:,dim:-1], a_min=-1, a_max=0)

    return flame_transform_vec

def flame_to_arkit_vector_local(flame_vector):
    """
    expecting a 2D array containing flame parameters
    :param flame_vector: 1x56 vector where 1-50 are flame shapekey values and 50-56 are pose information
    :pose angles are in axis angles, it needs to be converted to Euler anles
    :return: json_dic containing Apple ARKit converted shapekey (beaware this is for Charamel character
    so naming might slightly differ)
    """

    json_exp = {}
    json_pose = {}
    json_dic = {}
    flame_arr = np.array(flame_vector, dtype=np.float32)



    # take first flame parameters. we have only defined for charamel vectors for 20
    flame_exp = flame_arr[:,0:20]
    flame_pose = flame_arr[:,50:56]


    # pose
    glob_euler_xyz, jaw_euler_xyz = angle_axis_to_eurler_bactch(flame_pose)

    # transform expressions
    f_tranformed_vector = flame_transform_vector_to_ar(flame_exp)
    arkit_arr = np.matmul(f_tranformed_vector, transform_matrx)


    # normalise the ARKit values: clip values between 0,1
    ex_value_arr= arkit_arr.clip(0, 1)

    # create a json with names and values of sk
    for j in range(0,ex_value_arr.shape[0]):
        tmp_exp= {}
        tmp_pose= {}
        for i, keys in enumerate(shape_key_names):
            tmp_exp[keys] = float(ex_value_arr[j,int(i)])
        # for k, keys in enumerate(pose_names):
        #     tmp_pose[keys] = float(flame_pose[j,int(k)])
        tmp_pose["globx"] = float(glob_euler_xyz[j,0])
        tmp_pose["globy"] = float(glob_euler_xyz[j,1])
        tmp_pose["globz"] = float(glob_euler_xyz[j,2])

        tmp_pose["jawx"] = float(jaw_euler_xyz[j, 0])
        tmp_pose["jawy"] = float(jaw_euler_xyz[j, 1])
        tmp_pose["jawz"] = float(jaw_euler_xyz[j, 2])


        json_dic[j] = {'expressions':tmp_exp, 'poses':tmp_pose }


    return json_dic

def flame_to_arkit_vector(flame_vector, ignore_fr_from_start, logger= None):
    """
    expecting a 2D array containing flame parameters
    :param flame_vector: 1x56 vector where 1-50 are flame shapekey values and 50-56 are pose information
    :pose angles are in axis angles, it needs to be converted to Euler anles
    :return: json_dic containing Apple ARKit converted shapekey (beaware this is for Charamel character
    so naming might slightly differ)
    """
    process_start = time.time()
    json_exp = {}
    json_pose = {}
    json_dic = {}
    flame_arr = np.array(flame_vector, dtype=np.float32)



    # take first flame parameters. we have only defined for charamel vectors for 20
    flame_exp = flame_arr[ignore_fr_from_start:,0:20]
    flame_pose = flame_arr[ignore_fr_from_start:,50:56]


    # pose
    glob_euler_xyz, jaw_euler_xyz = angle_axis_to_eurler_bactch(flame_pose)
    # transform expressions
    f_tranformed_vector = flame_transform_vector_to_ar(flame_exp)
    arkit_arr = np.matmul(f_tranformed_vector, transform_matrx)


    # normalise the ARKit values: clip values between 0,1
    ex_value_arr= arkit_arr.clip(0, 1)

    # create a json with names and values of sk
    for j in range(0,ex_value_arr.shape[0]):
        tmp_exp= {}
        tmp_pose= {}
        for i, keys in enumerate(shape_key_names):
            tmp_exp[keys] = float(ex_value_arr[j,int(i)])
        # for k, keys in enumerate(pose_names):
        #     tmp_pose[keys] = float(flame_pose[j,int(k)])
        tmp_pose["globx"] = float(glob_euler_xyz[j,0])
        tmp_pose["globy"] = float(glob_euler_xyz[j,1])
        tmp_pose["globz"] = float(glob_euler_xyz[j,2])

        tmp_pose["jawx"] = float(jaw_euler_xyz[j, 0])
        tmp_pose["jawy"] = float(jaw_euler_xyz[j, 1])
        tmp_pose["jawz"] = float(jaw_euler_xyz[j, 2])


        json_dic[j] = {'expressions':tmp_exp, 'poses':tmp_pose }

    logger.info(f"Process:{time.time() - process_start}")
    return json_dic

if __name__ == "__main__":
    #parent = Path("/Users/deep/Documents/GitHub/Character/character/fastApi_backend/data/facial_expression/")
    #print(multi_array)
    #arkit_json = flame_to_arkit_vector(multi_array)
    #print(arkit_json)
    with open("/Users/deep/Documents/GitHub/Character/character/fastApi_backend/data/incoming_data/incoming_data.json") as jf:
        charamel_json = json.load(jf)

    for k in charamel_json:
       for child_k in charamel_json[k]:
           for keys in charamel_json[k][child_k]:
               charamel_json[k][child_k][keys] = 0.0
               #print(keys)
           # for exp in child_k["expressions"]:
           #      print(exp)

    print(charamel_json)

    with open("/Users/deep/Documents/GitHub/Character/character/fastApi_backend/data/neutral_pose.json", "w") as np_js:
        json.dump(charamel_json,np_js)








