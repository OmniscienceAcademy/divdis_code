from torch.utils.data import DataLoader
from PIL import Image
import os
from torchvision import transforms
import torch
from defaults import DATA_PATH
from glob import glob
import os
from defaults import *
import random
import json
import torch as t

DATA_PATH = DATA_PATH if len(glob(DATA_PATH)) > 0 else os.path.join(os.getcwd(), "data")


class StackedImg:
    """
    A collection of stacked images
    """

    # filenames : an iterable of filenames for the stacked images
    def __init__(self, *filenames):
        self.filenames = filenames

    # returns the stacked image as a PIL Image
    def open(self):
        return filenames_to_img(*self.filenames)

    # save a stacked image to save_dr
    def save(self, save_dr):
        img = self.open()
        img.save(save_dr)

    # for use when making a DataLoader of labeled StackedImgs
    # batch is a list of tuples (StackedImg, tensor1, tensor2, ..., tensorN)
    def _collator(batch):
        N = len(batch[0])
        return (
            [x[0] for x in batch],
            *(t.stack([x[i] for x in batch], dim=0) for i in range(1, N)),
        )


class Container:
    """
    a class for holding source (labeled) and target (unlabeled) datasets along with
    metadata about them
    """

    def __init__(
        self,
        dr,
        labeled_mix_rate=-1,
        unlabeled_mix_rate=0.5,
        labeled_batch_size=64,
        unlabeled_batch_size=64,
        vt_batch_size=64,
    ):

        if dr is None:
            self.dr = None
        else:
            self.dr = os.path.join(DATA_PATH, dr)

            # load information from metadata.json
            try:
                f = open(os.path.join(self.dr, "metadata.json"))
            except:
                raise ValueError(
                    "Either the data path is wrong, or the data folder is missing a metadata file"
                )
            metadata = json.load(f)
            self.classes = metadata["classes"]  # list of classes for each feature
            self.n_features = metadata["n_features"]  # number of features
            self.true_feat_idx = metadata[
                "true_feat_idx"
            ]  # index of the ground truth feature
            # self.fld[folder_name] gives a list of labels for each feature for the files in given folder
            self.fld = metadata["folder_label_dict"]

        # data needed to form source and target datasets
        self.mix_rate = {"labeled": labeled_mix_rate, "unlabeled": unlabeled_mix_rate}
        self.batch_size = {
            "labeled": labeled_batch_size,
            "unlabeled": unlabeled_batch_size,
            "val": vt_batch_size,
            "test": vt_batch_size,
        }
        # when needed, self.dataloaders will be populated with the labeled, unlabeled, val, and test dataloaders
        self.dataloaders = {}

    # luvt should be 'labeled', 'unlabeled', 'val', or 'test'
    def get_dataloader(self, luvt):
        assert luvt in [
            "labeled",
            "unlabeled",
            "val",
            "test",
        ], "Argument should be 'labeled', 'unlabeled', 'val', or 'test'"
        if luvt in self.dataloaders:
            return self.dataloaders[luvt]

        data = {}
        for dc in ["diag", "cross"]:
            folders = glob(os.path.join(self.dr, luvt, dc, "*"))
            data[dc] = []
            for folder in folders:
                labels = self.fld[os.path.basename(folder)]
                labels = [self.classes[i].index(cl) for i, cl in enumerate(labels)]
                labels = t.LongTensor(labels)
                true_label = labels[self.true_feat_idx]
                data[dc] += [
                    (StackedImg(file), true_label, labels)
                    for file in glob(os.path.join(folder, "*"))
                ]

        if luvt in ["val", "test"]:
            collated_data = []
            for dc in data:
                collated_data += data[dc]
            drop_last = False
        else:  # in the labeled and unlabeled cases, mix the data according to the mix rate
            r = self.mix_rate[luvt]
            if r == -1:  # take all data if mix rate is -1
                collated_data = data["diag"] + data["cross"]
            elif 0 <= r and r <= 1:
                n = min(len(data["diag"]), len(data["cross"]))
                n_diag, n_cross = int((1 - r) * n), int(r * n)
                collated_data = random.sample(data["diag"], n_diag) + random.sample(
                    data["cross"], n_cross
                )
            else:
                raise ValueError("Mix rate must be in [0,1] or == -1")
            drop_last = True

        try:
            dl = DataLoader(
                collated_data,
                self.batch_size[luvt],
                shuffle=True,
                collate_fn=StackedImg._collator,
                drop_last=drop_last,
            )
        except:
            raise ValueError(
                "Something went wrong while making the DataLoader, probably because "
                "it was fed an empty dataset. Did you use mix_rate=0 when you meant mix_rate=-1?"
            )
        self.dataloaders[luvt] = dl
        return dl

    # drs : a nonempty list of directories
    # classes : list whose ith entry is a list of class names for classes in drs[i]
    # label_dr_idx : the index in drs of the directory to treat as ground truth for labeling
    def stack_from_drs(
        drs,
        classes,
        true_feat_idx,
        train_n,
        val_n,
        test_n,
        train_mix_rates=[0.0, 0.5],
        val_mix_rates=[0.0, 0.5],
        test_mix_rates=[0.0, 0.5],
        source_batch_size=128,
        target_batch_size=128,
    ):
        # CURRENTLY DEPRICATED TODO
        n_drs = len(drs)
        if len(classes) != n_drs:
            raise ValueError
        n_classes = len(classes[0])
        for class_list in classes:
            if len(class_list) != n_classes:
                raise ValueError("All class lists should have the same length")

        out = Container(
            None,
            train_mix_rates=train_mix_rates,
            val_mix_rates=val_mix_rates,
            test_mix_rates=test_mix_rates,
            source_batch_size=source_batch_size,
            target_batch_size=target_batch_size,
        )
        out.classes = classes
        out.n_features = n_drs
        out.true_feat_idx = true_feat_idx

        n = {"train": train_n, "val": val_n, "test": test_n}

        for tvt in ["train", "val", "test"]:
            # gather lists of files for each class
            filelists = [[[] for _ in range(n_classes)] for _ in range(n_drs)]
            for i, dr in enumerate(drs):
                for j, cl in enumerate(classes[i]):
                    path = os.path.join(DATA_PATH, dr, tvt, cl, "*")
                    filelists[i][j] = glob(path)

            out.data[tvt] = {}

            # construct diag data

            out.data[tvt]["diag"] = []
            while len(out.data[tvt]["diag"]) < n[tvt]:
                cl_id = random.randint(0, n_classes - 1)
                datapoint = tuple(
                    random.choice(filelists[i][cl_id]) for i in range(n_drs)
                )
                out.data[tvt]["diag"].append(
                    (datapoint, cl_id, t.LongTensor([cl_id] * n_drs))
                )

            # construct cross data
            out.data[tvt]["cross"] = []
            while len(out.data[tvt]["cross"]) < n[tvt]:
                cl_ids = random.choices(range(n_classes), k=n_drs)
                if all(idx == cl_ids[0] for idx in cl_ids):
                    pass
                else:
                    datapoint = tuple(
                        random.choice(filelists[i][cl_ids[i]]) for i in range(n_drs)
                    )
                    out.data[tvt]["cross"].append(
                        (datapoint, cl_ids[true_feat_idx], t.LongTensor(cl_ids))
                    )

        return out

    # save data to save_dr
    def save(self, save_dr):
        # CURRENTLY DEPRICATED TODO
        save_dr = os.path.join(DATA_PATH, save_dr)
        os.mkdir(save_dr)

        # save image files
        folder_label_dict = {}
        for tvt in ["train", "val", "test"]:
            self._load_data(tvt)
            os.mkdir(os.path.join(save_dr, tvt))
            for dc in ["diag", "cross"]:
                os.mkdir(os.path.join(save_dr, tvt, dc))
                file_count = {}
                for filenames, _, labels in self.data[tvt][dc]:
                    img = filenames_to_img(*filenames)
                    cl_name = self._class_name_from_labels(labels)
                    if cl_name not in folder_label_dict:
                        folder_label_dict[cl_name] = [
                            self.classes[i][label] for i, label in enumerate(labels)
                        ]
                    if (
                        cl_name not in file_count
                    ):  # if this is the first image of its class for this tvt/dc
                        os.mkdir(os.path.join(save_dr, tvt, dc, cl_name))
                        file_count[cl_name] = 0
                    else:
                        file_count[cl_name] += 1
                    img.save(
                        os.path.join(
                            save_dr,
                            tvt,
                            dc,
                            cl_name,
                            cl_name + "_{:04d}".format(file_count[cl_name]),
                        )
                        + ".png"
                    )

        # save metadata
        metadata = {
            "n_features": self.n_features,
            "true_feat_idx": self.true_feat_idx,
            "classes": self.classes,
            "folder_label_dict": folder_label_dict,
        }
        with open(os.path.join(save_dr, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    # labels : a nonempty list (or tensor) of class ids
    # concatinate the corresponding class names with separating -'s
    def _class_name_from_labels(self, labels):
        return "-".join([self.classes[i][label] for i, label in enumerate(labels)])


def directory_to_filepath_list(dir, ext="jpg"):
    """
    returns a list of all filepaths with a given extension (by default .jpg) in a given directory
    """
    filepaths = glob(dir + f"{os.sep}*.{ext}")
    return filepaths


"""
def directory_to_resized_image_list(dir, ext = 'jpg', new_width, new_height):

    # returns a list of resized images (resized to given new dimensions) with a given extension (by default .jpg) in a given directory - NEEDS TESTING!

    imgpathlist = directory_to_filepath_list(dir, ext)
    resized_imglist = []
    for innum, path in enumerate(imgpathlist):
        img = Image.open(path)
        resized_imglist.append(img.resize(new_width,new_height))
    return resized_imglist
 """


def tensorise_image_list(imglist):
    """
    takes a list of images and returns a corresponding list of their tensorisations
    """
    tensorised_imglist = [transforms.ToTensor()(image) for image in imglist]
    return tensorised_imglist


def tensorstack_image_list(imgs):
    """
    takes a list of images, tensorises them and stacks all the tensors into a single tensor along a new dimension, returns that
    """
    if len(imgs) > 0:
        stacked_imgs = torch.stack(imgs)
    else:
        stacked_imgs = torch.empty(
            0
        )  # if ims is an empty list, return an empty tensor (shape (0))
    return stacked_imgs


def filenames_to_img(*filenames):
    return get_concat_v([Image.open(file) for file in filenames])


def filenames_to_tensor(*filenames):
    return transforms.ToTensor()(filenames_to_img(*filenames))


def imgs_to_tensor_from_dir(dir, proportion_of_images=1.0):
    """
    takes a pathname of a directory and returns a single tensor which is a stack (but not in the visual sense!)
    of a given proportion (by default all) of the (3d) tensorisations of images with a given extension (by default jpg)
    which are in that directory
    """
    images = []
    imagepaths = directory_to_filepath_list(dir)
    random.shuffle(imagepaths)
    number_of_images = len(imagepaths)
    images_to_take = round(
        number_of_images * proportion_of_images
    )  # defaults to number_of_images
    for imnum, img_path in enumerate(imagepaths):
        if imnum < images_to_take:
            images.append(filename_to_tensor(img_path))
    return tensorstack_image_list(images)


def get_concat_v(ims: [Image.Image]) -> Image.Image:
    """
    visually stacks the images in the (list) argument to create and return a single image
    """
    heights = [0]
    current_height = 0
    max_width = 0
    for im in ims:
        current_height += (
            im.height
        )  # cumulatively adding up heights of all the images in the list 'ims'
        heights.append(
            current_height
        )  # the list 'heights' will be used as a guide to the horizontal levels at which these images will be pasted
        max_width = max(
            max_width, im.width
        )  # 'max_width' is updated if a "widest yet" image is newly encountered
    dst = Image.new(
        "RGB", (max_width, current_height)
    )  # initialises a blank RGB image 'dst' of the appropriate dimensions
    for i, im in enumerate(ims):
        dst.paste(
            im, (0, heights[i])
        )  # pastes the images sequentially into dst at their appropriate heights
    dst = dst.convert("RGB")  # converts resulting image into RGB mode
    return dst


def stacked_imgs_to_tensor_from_dirs(dirs, proportion_of_images=1.0):
    """
    returns a single tensor encoding a list of visually stacked images,
    where each image in each stack is from a different directory in the list 'dirs' (and all have the same index)
    """
    images = []
    dir_sizes = [len(directory_to_filepath_list(dirs[d])) for d in range(len(dirs))]
    number_of_images = min(
        dir_sizes
    )  # whichever of the directories in dirs has the least image files, use that many
    images_to_take = round(
        number_of_images * proportion_of_images
    )  # 'images_to_take' defaults to 'number_of_images'
    imgpathlists = (
        []
    )  # 'imgpathlists' is going to be a list of (ususally 2) lists of imagepaths, each of which contains pathnames of all image files in one of the (usually 2) directories in dir
    height = len(
        dirs
    )  # this 'height' is the number of images that are going to get visuallly stacked
    for dir in dirs:
        imgpathlists.append(
            directory_to_filepath_list(dir)
        )  # add lists of imagepaths, one directory at at time
        random.shuffle(
            imgpathlists[-1]
        )  # the last (usually second) of these image pathlists lists gets randomised
    for imnum, img_paths in enumerate(
        zip(*imgpathlists)
    ):  # zip(*filelists) will be a list of lists of (usually 2) image pathnames, one from each list in 'imgpathlists', occupying corresponding positions
        # 'enumerate' then makes a list of ordered pairs [(0,.), (1,.), (2,.),...], where the second member of each is an element of zip(*files)
        if imnum < images_to_take:  # loop through index numbers
            raw_images = (
                []
            )  # for each one, starting with a new empty list 'raw_images'...
            for h in range(height):
                raw_images.append(
                    Image.open(img_paths[h])
                )  # ...take the ordered pair from the above list with index number 'imnum',
                # and add to 'raw_images' all (by default 2) of the actual images whose pathnames
                # are listed in the second entry of the ordered pair
                # this is the raw visual data ready to get stacked
            images.append(
                get_concat_v(raw_images)
            )  # now use 'get_concat_v' to vertically stack the images in 'basic_images' and append the result to the list 'images'
    if len(images) > 0:
        stacked_imgs = torch.stack(
            tensorise_image_list(images)
        )  # torch.stack concatenates a sequence of tensors along a new dimension
        # they must be of the same size
        # so 'stacked_imgs' is a single 4-d tensor built from the whole set of stacked images
    else:
        stacked_imgs = torch.empty(0)
    return stacked_imgs  # if, for some reason there are no tensorised stacked images in the list 'images'
    # then return an empty 1-d tensor of shape (0)
