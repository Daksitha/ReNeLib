from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
# dkw
from gdl.datasets.ImageTestDataset import TestData
from gdl.utils.FaceDetector import FAN
from gdl.datasets.ImageDatasetHelpers import bbox2point
import cv2
import numpy as np
from skimage.transform import rescale, estimate_transform, warp
import scipy
import torch

import time
import tempfile
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default=str(
        Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4"),
                        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output",
                        help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA',
                        help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str,
                        default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False,
                        help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail',
                        choices=["geometry_detail", "geometry_coarse", "output_images_detail", "output_images_coarse"],
                        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None,
                        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
                             "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    parser.add_argument('--cat_dim', type=int, default=0,
                        help="The result video will be concatenated vertically if 0 and horizontally if 1")
    parser.add_argument('--include_transparent', type=bool, default=False,
                        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")

    #######Image####
    # add the input folder arg
    parser.add_argument('--imgs_input_folder', type=str, default=str(
        Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/images/affectnet_test_examples"))
    parser.add_argument('--imgs_output_folder', type=str, default="image_output",
                        help="Output folder to save the results to.")
    args = parser.parse_args()
    DATA_DIR = Path("/media/daksitha/C648B0E148B0D183/Daksitha_thesis/KASSEL_DATA_ML_READY/dataset/OPD_103-00226-01923")
    for vfiles in list(DATA_DIR.glob("*/*.mp4")):
        # print(vfiles.parent.name)
        extractor = FLAME_Video(input_video=str(vfiles),
                                output_folder=str(Path(DATA_DIR / vfiles.parent.name / "EMOCA_Output")))
        extractor.prepare_dataset_from_video()
        extractor.prepare_models()
        extractor.run(save_mesh=False, save_images_tag=False, save_codes_tag=False, extract_flame=True)

    # flam_img = FLAME_Images()
    # flam_img.prepare_models()
    # flam_img.prepare_dataset_from_images()
    # flam_img.run()

    # start_time = time.time()
    web_flame = FLAME_Webcam()
    web_flame.prepare_models()
    web_flame.run()
    # print("--- %s seconds ---" % (time.time() - start_time))


class FLAME_Webcam:
    def __init__(self, iscrop=True, crop_size=224, scale=1.25, face_detector='fan',
                 scaling_factor=1.0, max_detection=None, output_folder="webcam_output",
                 model_name='EMOCA',
                 path_to_models=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
                 ):
        self.path_to_models = path_to_models
        self.mode = 'detail'  # mode = 'coarse'
        self.model_name = model_name
        self.output_folder = output_folder
        self.max_detection = max_detection
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size

        self.cap = cv2.VideoCapture(0)

        if face_detector == 'fan':
            self.face_detector = FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def _get_face_detected_torch_image(self, webcam_frame, frame_number):
        imagepath = "webcam"
        imagename = str(frame_number)
        image = np.array(webcam_frame)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor, self.scaling_factor, 1)) * 255.

        h, w, _ = image.shape
        if self.iscrop:
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 1:
                print('no face detected!')
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
                old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            else:
                if self.max_detection is None:
                    bbox = bbox[0]
                    left = bbox[0]
                    right = bbox[2]
                    top = bbox[1]
                    bottom = bbox[3]
                    old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
                else:
                    old_size, center = [], []
                    num_det = min(self.max_detection, len(bbox))
                    for bbi in range(num_det):
                        bb = bbox[0]
                        left = bb[0]
                        right = bb[2]
                        top = bb[1]
                        bottom = bb[3]
                        osz, c = bbox2point(left, right, top, bottom, type=bbox_type)
                    old_size += [osz]
                    center += [c]

            if isinstance(old_size, list):
                size = []
                src_pts = []
                for i in range(len(old_size)):
                    size += [int(old_size[i] * self.scale)]
                    src_pts += [np.array(
                        [[center[i][0] - size[i] / 2, center[i][1] - size[i] / 2],
                         [center[i][0] - size[i] / 2, center[i][1] + size[i] / 2],
                         [center[i][0] + size[i] / 2, center[i][1] - size[i] / 2]])]
            else:
                size = int(old_size * self.scale)
                src_pts = np.array(
                    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                     [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        image = image / 255.
        if not isinstance(src_pts, list):
            DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
            dst_image = dst_image.transpose(2, 0, 1)
            return {'image': torch.tensor(dst_image).float(),
                    'image_name': imagename,
                    'image_path': imagepath,
                    # 'tform': tform,
                    # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    }
        else:
            DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            dst_images = []
            for i in range(len(src_pts)):
                tform = estimate_transform('similarity', src_pts[i], DST_PTS)
                dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
                dst_image = dst_image.transpose(2, 0, 1)
                dst_images += [dst_image]
            dst_images = np.stack(dst_images, axis=0)

            imagenames = [imagename + f"{j:02d}" for j in range(dst_images.shape[0])]
            imagepaths = [imagepath] * dst_images.shape[0]
            return {'image': torch.tensor(dst_images).float(),
                    'image_name': imagenames,
                    'image_path': imagepaths,
                    # 'tform': tform,
                    # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                    }

    def prepare_models(self):
        ## 2) Load the model
        self.emoca, conf = load_model(self.path_to_models, self.model_name, self.mode)
        self.emoca.cuda()
        self.emoca.eval()


    def run(self, save_mesh=False, save_images_tag=False, save_codes_tag=False):
        ## run the webcam

        # Loop until you hit the Esc key
        # if DEBUG_MODE = True:

        frame_count = 0
        while True:
            # Capture the current frame
            ret, frame = self.cap.read()
            # im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                print("failed to grab frame")
                break

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            batch = self._get_face_detected_torch_image(frame, frame_count)

            vals, visdict = test(self.emoca, batch)
            # name = f"{i:02d}"
            current_bs = batch["image"].shape[0]

            for j in range(current_bs):
                name = batch["image_name"][j]

                sample_output_folder = Path(self.output_folder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)

                if save_mesh:
                    save_obj(self.emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
                if save_images_tag:
                    save_images(self.output_folder, name, visdict, with_detection=True, i=j)
                if save_codes_tag:
                    save_codes(Path(self.output_folder), name, vals, i=j)

            print(f"frame_count: {frame_count}")
            frame_count += 1
        self.cam.release()

        cv2.destroyAllWindows()
        print("Done")


class FLAME_Images:
    def __init__(self,
                 input_folder=str(
                     Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/images/affectnet_test_examples"),
                 output_folder="image_output",
                 model_name='EMOCA',
                 path_to_models=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"),
                 flame_sequance_fames=64
                 ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_name = model_name
        self.path_to_models = path_to_models
        self.mode = 'detail'  # mode = 'coarse'

        self.flame_sequence_size = flame_sequance_fames
        self.num_features = 184
        self.bin_counter = 0
        self.start_frame = None
        self.bins = None

    def prepare_models(self):
        ## 2) Load the model
        self.emoca, conf = load_model(self.path_to_models, self.model_name, self.mode)
        self.emoca.cuda()
        self.emoca.eval()

    def prepare_dataset_from_images(self):
        # 2) Create a dataset
        self.dataset = TestData(self.input_folder, face_detector="fan", max_detection=20)

    def _flame_extractor(self, vals, i=None):
        """
        per frame save flame if it is a video
        vals: torch tensor containing flame parameters
        i: frame
        """
        if i is None:
            exprnpy = vals["expcode"].detach().cpu().numpy()
            posenpy = vals["posecode"].detach().cpu().numpy()
            detailnpy = vals["detailcode"].detach().cpu().numpy()

            flamefeatures = np.concatenate((exprnpy, posenpy, detailnpy), axis=None)
            # print(f"shapes of flame code: exprnpy:{exprnpy.shape},posenpy:{posenpy.shape}, detailnpy:{detailnpy.shape}  ")
            # print(f" flame code: exprnpy:{exprnpy},posenpy:{posenpy}, detailnpy:{detailnpy}  ")
            return flamefeatures
        else:
            exprnpy = vals["expcode"][i].detach().cpu().numpy()
            posenpy = vals["posecode"][i].detach().cpu().numpy()
            detailnpy = vals["detailcode"][i].detach().cpu().numpy()

            flamefeatures = np.concatenate((exprnpy, posenpy, detailnpy), axis=None)
            # print(f"shapes of flame code: exprnpy:{exprnpy.shape},posenpy:{posenpy.shape}, detailnpy:{detailnpy.shape}  ")
            # print(f" flame code: exprnpy:{exprnpy},posenpy:{posenpy}, detailnpy:{detailnpy}  ")
            # print(f"flamefeatures shape: {flamefeatures.shape} \n flamefeatures:{flamefeatures}")
            return flamefeatures

    def save_flame_sequence(self, name=None, vals=None, i=None):
        """
        vals: torch tensor containing flame parameters
        i: frame
        name:  folder name, usually used as the frame number

        NOTE:extracted features across the video into a (1 x T x (50+6)) matrix and save to
        in concatenating the features, expression comes first.
        """

        flame_per_fame = self._flame_extractor(vals, i)
        if i is None:
            # images
            np.save(self.output_folder / name / f"flame_feature_frame.npy", flame_per_fame)
        else:
            print(flame_per_fame.shape[0])
            # if self.num_features == flame_per_fame.shape[0]:
            #
            #     if self.start_frame is None:
            #         self.start_frame = name
            #         self.bins = np.zeros(
            #             (1, int(self.flame_sequence_size), int(self.num_features)))  # (Exp:50+Pose:6+detail:128)
            #
            #     self.bins[:, self.bin_counter, :] = flame_per_fame
            #     self.bin_counter += 1
            #
            #     if self.bin_counter == self.flame_sequence_size:
            #         final_out_folder = Path(self.output_folder) / "flame_sequances"
            #         final_out_folder.mkdir(parents=True, exist_ok=True)
            #         np.save(final_out_folder / f"flame_sequance_{self.start_frame}_{name}.npy", flame_per_fame)
            #         # reset bin
            #
            #         self.bin_counter = 0
            #         self.start_frame = None
            #
            # else:
            #     raise NameError(f"num_features {self.num_features} not equal to incoming \
            #     feature array {flame_per_fame.shape}")

    def run(self, save_mesh=False, save_images_tag=False, save_codes_tag=False, extract_flame=True):
        ## 4) Run the model on the data
        for i in auto.tqdm(range(len(self.dataset))):
            batch = self.dataset[i]
            vals, visdict = test(self.emoca, batch)
            # name = f"{i:02d}"
            current_bs = batch["image"].shape[0]

            for j in range(current_bs):
                name = batch["image_name"][j]

                sample_output_folder = Path(self.output_folder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)

                if save_mesh:
                    save_obj(self.emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
                if save_images_tag:
                    save_images(self.output_folder, name, visdict, with_detection=True, i=j)
                if save_codes_tag:
                    save_codes(Path(self.output_folder), name, vals, i=j)
                if extract_flame:
                    self.save_flame_sequence(name=name, vals=vals, i=j)

        print("Done")


#### Video Processing #####
# model_nam: Name of the model to use. Currently EMOCA or DECA are available.
# image_type: ["geometry_detail", "geometry_coarse", "output_images_detail", "output_images_coarse"]
#           Which image to use for the reconstruction video
# processed_subfolder "If you want to resume previously interrupted computation over a video, make sure you specify" \
#            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'"
# cat_dim: "The result video will be concatenated vertically if 0 and horizontally if 1"
# include_transparent: Apart from the reconstruction video, also a video with the transparent mesh will be added
class FLAME_Video:
    def __init__(self, path_to_models=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"),
                 input_video=str(
                     Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/videos/82-25-854x480_affwild2.mp4"),
                 output_folder="video_output",
                 model_name='EMOCA',
                 image_type='geometry_detail',
                 cat_dim=0,
                 include_transparent=False,
                 processed_subfolder=None):

        self.num_features = None
        self.flame_outfolder = None
        self.dm = None
        self.outfolder = None
        self.dl = None
        self.emoca = None
        self.path_to_models = path_to_models
        self.input_video = input_video
        self.output_folder = output_folder
        self.model_name = model_name

        self.image_type = image_type
        self.cat_dim = cat_dim
        self.include_transparent = bool(include_transparent)
        print("Include transparent:", self.include_transparent)
        self.processed_subfolder = processed_subfolder

        print("Path to models " + self.path_to_models)
        self.mode = 'detail'  # mode = 'coarse'

    def prepare_dataset_from_video(self):
        ## 1) Process the video - extract the frames from video and detected faces
        # processed_subfolder="processed_2022_Jan_15_02-43-06"
        # processed_subfolder=None
        self.dm = TestFaceVideoDM(self.input_video, self.output_folder, processed_subfolder=self.processed_subfolder,
                                  batch_size=4, num_workers=4)
        self.dm.prepare_data()
        self.dm.setup()
        processed_subfolder = Path(self.dm.output_dir).name
        self.outfolder = str(
            Path(self.output_folder) / processed_subfolder / Path(self.input_video).stem / "results" / self.model_name)
        self.flame_outfolder = str(
            Path(self.output_folder) / processed_subfolder / Path(
                self.input_video).stem / "results" / "Learn2ListenData")
        ##Get the data loadeer with the detected faces
        self.dl = self.dm.test_dataloader()

    def prepare_models(self):
        ## 2) Load the model
        self.emoca, conf = load_model(self.path_to_models, self.model_name, self.mode)
        self.emoca.cuda()
        self.emoca.eval()

    def _flame_extractor(self, vals, i=None):
        # NOTE:extracted features across the video into a (1 x T x (50+6)) matrix and save to
        # in concatenating the features, expression comes first.
        if i is None:
            exprnpy = vals["expcode"].detach().cpu().numpy()
            posenpy = vals["posecode"].detach().cpu().numpy()
            detailnpy = vals["detailcode"].detach().cpu().numpy()

            flamefeatures = np.concatenate((exprnpy, posenpy, detailnpy), axis=None)
            # print(f"shapes of flame code: exprnpy:{exprnpy.shape},posenpy:{posenpy.shape}, detailnpy:{detailnpy.shape}  ")
            # print(f" flame code: exprnpy:{exprnpy},posenpy:{posenpy}, detailnpy:{detailnpy}  ")
            return flamefeatures
        else:
            exprnpy = vals["expcode"][i].detach().cpu().numpy()
            posenpy = vals["posecode"][i].detach().cpu().numpy()
            detailnpy = vals["detailcode"][i].detach().cpu().numpy()

            flamefeatures = np.concatenate((exprnpy, posenpy, detailnpy), axis=None)
            # print(f"shapes of flame code: exprnpy:{exprnpy.shape},posenpy:{posenpy.shape}, detailnpy:{detailnpy.shape}  ")
            # print(f" flame code: exprnpy:{exprnpy},posenpy:{posenpy}, detailnpy:{detailnpy}  ")
            # print(f"flamefeatures shape: {flamefeatures.shape} \n flamefeatures:{flamefeatures}")

            return flamefeatures

    def save_flame_sequence(self, name=None, vals=None, i=None):
        flame_per_fame = self._flame_extractor(vals, i)
        if i is None:
            np.save(self.output_folder / name / f"flame_feature_frame.npy", flame_per_fame)
        else:
            print(flame_per_fame.shape[0])
            # if self.num_features == flame_per_fame.shape[0]:
            #
            #     if self.start_frame is None:
            #         self.start_frame = name
            #         self.bins = np.zeros(
            #             (1, int(self.flame_sequence_size), int(self.num_features)))  # (Exp:50+Pose:6+detail:128)
            #
            #     self.bins[:, self.bin_counter, :] = flame_per_fame
            #     self.bin_counter += 1
            #
            #     if self.bin_counter == self.flame_sequence_size:
            #         final_out_folder = Path(self.flame_outfolder) / "flame_sequances"
            #         final_out_folder.mkdir(parents=True, exist_ok=True)
            #         np.save(final_out_folder / f"flame_sequance_{self.start_frame}_{name}.npy", flame_per_fame)
            #         # reset bin
            #
            #         self.bin_counter = 0
            #         self.start_frame = None
            #
            # else:
            #     raise NameError(f"num_features {self.num_features} not equal to incoming \
            #     feature array {flame_per_fame.shape}")

    def run(self, save_mesh=False, save_images_tag=True, save_codes_tag=False, extract_flame=True):

        ## 4) Run the model on the data
        for j, batch in enumerate(auto.tqdm(self.dl)):
            current_bs = batch["image"].shape[0]
            img = batch
            vals, visdict = test(self.emoca, img)
            for i in range(current_bs):
                # name = f"{(j*batch_size + i):05d}"
                name = batch["image_name"][i]

                sample_output_folder = Path(self.outfolder) / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)

                if save_mesh:
                    save_obj(self.emoca, str(sample_output_folder / "mesh_coarse.obj"), vals, i)
                if save_images_tag:
                    save_images(self.outfolder, name, visdict, i)
                if save_codes_tag:
                    save_codes(Path(self.outfolder), name, vals, i)
                if extract_flame:
                    self.save_flame_sequence(name=name, vals=vals, i=j)

    def reconstruct_video_with_overlay(self):
        ## 5) Create the reconstruction video (reconstructions overlayed on the original video)
        self.dm.create_reconstruction_video(0, rec_method=self.model_name, image_type=self.image_type, overwrite=True,
                                            cat_dim=self.cat_dim, include_transparent=self.include_transparent)
        print("Done creating the reconstruction video ")


if __name__ == '__main__':
    main()
