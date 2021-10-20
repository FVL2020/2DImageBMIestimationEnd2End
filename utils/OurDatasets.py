import torch.utils.data as data
import torch

from utils import *
import os
import re
import cv2
import json
from utils.skeleton_feat import *

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_dataloader(batch_size, args, ):
    train_dataset = OurDatasets(args.root, 'Image_train', mode=args.datasetmode, set=args.set, kpts_fea=args.kpts)
    test_dataset = OurDatasets(args.root, 'Image_test', mode=args.datasetmode, set=args.set, kpts_fea=args.kpts,
                               partition='test')
    val_dataset = OurDatasets(args.root, 'Image_val', mode=args.datasetmode, set=args.set, kpts_fea=args.kpts,
                              partition='val')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    return train_loader, val_loader, test_loader


class OurDatasets(data.Dataset):
    def __init__(self, root, file, mode, set='Our', kpts_fea=False, partition='train', RGB2GRAY=False):
        self.file = os.path.join(root, file)
        self.img_names = os.listdir(self.file)
        # print(os.path.join(root, file[6:]+'_kpts.json'))
        # with open(os.path.join(root, file[6:] + '_kpts.json')) as f:
        #     self.img_kpts = json.load(f)
        self.mode = mode
        self.set = set
        if partition == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                Resize(IMG_SIZE),
                transforms.Pad(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                # argue
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5),
                # transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                Resize(IMG_SIZE),
                transforms.Pad(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                # transforms.Grayscale(),
                transforms.ToTensor(),
            ])

        self.kpts_fea = kpts_fea

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        global Skeletons
        img_name = self.img_names[idx]
        img_mask_name = 'Mask_' + img_name
        img = cv2.imread(os.path.join(self.file, img_name), flags=1)[:, :, ::-1]
        h, w, _ = img.shape

        if self.kpts_fea == True:
            img_size = 224
            kpts = self.img_kpts[img_name]
            m1 = stride_matrix(IMG_SIZE / h, (IMG_SIZE - w * IMG_SIZE / h) / 2)
            warped_kpts = warpAffineKpts([kpts], m1)
            m2 = stride_matrix(img_size / IMG_SIZE, (img_size - IMG_SIZE * img_size / IMG_SIZE) / 2)
            warped_kpts = warpAffineKpts(warped_kpts, m2)
            Skeletons = genSkeletons(warped_kpts, img_size, img_size, ).transpose(2, 0, 1)
            Skeletons = torch.from_numpy(Skeletons)
            Skeletons = torch.unsqueeze(torch.sum(Skeletons, 0), 0)

        if self.mode == '4C':
            img = self.transform(img)
            img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)
            img_mask = cv2.imread(os.path.join(self.file + '_mask', img_mask_name), flags=0)
            img_mask = self.transform(img_mask)
            img_c = torch.cat((img, img_mask), dim=0)

        elif self.mode == '3CWithMask':
            img_mask = cv2.imread(os.path.join(self.file + '_mask', img_mask_name))
            img_c = img * (img_mask // 255 == 0)

            # set background to white
            # img_mask = img_mask * (img_mask // 255 == 1)
            # img_c = img_c + img_mask

            img_c = self.transform(img_c)

            img_c = transforms.Normalize(IMG_MEAN, IMG_STD)(img_c)


        elif self.mode == '3C':
            img = self.transform(img)
            img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)
            img_c = img
        else:
            img_c = None

        if self.set == 'Our':
            ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
            sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
            BMI = torch.from_numpy(np.asarray((int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2))
        elif self.set == 'Author':
            ret = re.match(r"[a-zA-Z0-9]+_[a-zA-Z0-9]+__?(\d+)__?(\d+)__?([a-z]+)_*", img_name)
            height = float(ret.group(2)) * 0.0254
            weight = float(ret.group(1)) * 0.4536
            sex = (lambda x: x == 'false')(ret.group(3))
            BMI = weight / (height ** 2)
        else:
            sex, BMI = None, None

        if self.kpts_fea:
            return (img_c, Skeletons.float()), (sex, BMI)
        else:
            return img_c, (sex, BMI)

