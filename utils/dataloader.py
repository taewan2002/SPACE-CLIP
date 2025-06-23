# utils/dataloader.py

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 잘린 이미지 파일도 로드할 수 있도록 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _is_pil_image(img):
    """PIL Image 객체인지 확인"""
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    """Numpy ndarray 이미지인지 확인"""
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def preprocessing_transforms(mode):
    """전처리 파이프라인 생성"""
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

class DepthDataLoader(object):
    """
    데이터 로더를 생성하고 관리하는 메인 클래스.
    모드('train', 'online_eval', 'test')에 따라 적절한 DataLoader 인스턴스를 생성.
    """
    def __init__(self, args, mode, collate_fn=None):
        self.dataset = DataLoadPreprocess(args, mode,
                                          transform=preprocessing_transforms(mode))
        
        if mode == 'train':
            if args.distributed:
                self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            else:
                self.sampler = None

            self.data = DataLoader(self.dataset,
                                   args.batch_size,
                                   shuffle=(self.sampler is None),
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   sampler=self.sampler,
                                   collate_fn=collate_fn,
                                   drop_last=True)
        else: # 'online_eval' or 'test'
            self.sampler = None
            self.data = DataLoader(self.dataset,
                                   getattr(args, 'eval_batch_size', 1),
                                   shuffle=False,
                                   num_workers=getattr(args, 'eval_workers', 1),
                                   pin_memory=False,
                                   sampler=self.sampler,
                                   collate_fn=collate_fn)

def remove_leading_slash(s):
    """경로 문자열의 맨 앞 슬래시(/)나 역슬래시(\) 제거"""
    if s and (s[0] == '/' or s[0] == '\\'):
        return s[1:]
    return s

class DataLoadPreprocess(Dataset):
    """
    실제 데이터 로딩과 전처리를 담당하는 Dataset 클래스.
    파일이 없거나 유효하지 않으면 None을 반환.
    """
    def __init__(self, args, mode, transform=None):
        self.args = args
        self.mode = mode
        self.clip_input_size = getattr(args, 'clip_input_size', 224)
        self.target_height = getattr(args, 'input_height', 352)
        self.target_width = getattr(args, 'input_width', 704)
        
        if mode == 'online_eval':
            filenames_path = args.filenames_file_eval
        elif mode == 'train':
            filenames_path = args.filenames_file
        elif mode == 'test':
            filenames_path = getattr(args, 'filenames_file_test', args.filenames_file_eval)
        else:
            raise ValueError(f"Invalid mode '{mode}' for filenames.")

        try:
            with open(filenames_path, 'r') as f:
                self.filenames = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Filenames file not found at: {filenames_path}")

        self.transform = transform

    def __getitem__(self, idx):
        try:
            sample_line = self.filenames[idx]
            parts = sample_line.split()
            
            img_rel_path = parts[0]
            depth_rel_path = parts[1] if len(parts) > 1 else None
            focal = float(parts[2]) if len(parts) > 2 else 0.0

            pil_image, pil_depth_gt = self._load_images(img_rel_path, depth_rel_path)

            if pil_image is None:
                return None

            if self.mode == 'train':
                if pil_depth_gt is None:
                    return None
                
                image_processed_np, depth_np = self._preprocess_train(pil_image, pil_depth_gt)
                
                pil_for_clip_resize = Image.fromarray((image_processed_np * 255).astype(np.uint8))
                pil_for_clip_resized = pil_for_clip_resize.resize((self.clip_input_size, self.clip_input_size), Image.BICUBIC)
                image_for_clip_np = np.asarray(pil_for_clip_resized, dtype=np.float32) / 255.0
                
                sample = {
                    'image_clip': image_for_clip_np,
                    'image_orig': image_processed_np,
                    'depth': depth_np,
                    'focal': focal
                }
            else: 
                image_processed_np, depth_np, has_valid_depth = self._preprocess_eval_test(pil_image, pil_depth_gt)

                pil_for_clip_resize = Image.fromarray((image_processed_np * 255).astype(np.uint8))
                pil_for_clip_resized = pil_for_clip_resize.resize((self.clip_input_size, self.clip_input_size), Image.BICUBIC)
                image_for_clip_np = np.asarray(pil_for_clip_resized, dtype=np.float32) / 255.0

                sample = {
                    'image_clip': image_for_clip_np,
                    'image_orig': image_processed_np,
                    'focal': focal
                }
                if self.mode == 'online_eval':
                    sample['depth'] = depth_np if has_valid_depth and depth_np.size > 0 else np.array([])
                    sample['has_valid_depth'] = has_valid_depth
                    sample['image_path'] = img_rel_path
                    sample['depth_path'] = depth_rel_path if depth_rel_path else ""
            
            if self.transform:
                sample = self.transform(sample)
            return sample

        except Exception:
            return None

    def _load_images(self, img_rel_path, depth_rel_path):
        if self.mode == 'train':
            base_data_path, base_gt_path = self.args.data_path, self.args.gt_path
        elif self.mode == 'online_eval':
            base_data_path, base_gt_path = self.args.data_path_eval, self.args.gt_path_eval
        else: # 'test'
            base_data_path, base_gt_path = getattr(self.args, 'data_path_test', self.args.data_path), None

        image_path = os.path.join(base_data_path, remove_leading_slash(img_rel_path))
        try:
            pil_image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, IOError):
            pil_image = None

        pil_depth_gt = None
        if depth_rel_path and base_gt_path:
            depth_path = os.path.join(base_gt_path, remove_leading_slash(depth_rel_path))
            try: 
                pil_depth_gt = Image.open(depth_path)
            except (FileNotFoundError, IOError):
                pil_depth_gt = None
        
        return pil_image, pil_depth_gt

    def _apply_kb_crop(self, pil_image, pil_depth_gt=None):
        h, w = pil_image.height, pil_image.width
        th, tw = 352, 1216
        if w < tw or h < th:
            pil_image = pil_image.resize((max(tw, w), max(th, h)), Image.BICUBIC)
            if pil_depth_gt: pil_depth_gt = pil_depth_gt.resize((max(tw, w), max(th, h)), Image.NEAREST)
            h, w = pil_image.height, pil_image.width
        top = h - th
        left = int(round((w - tw) / 2.0))
        pil_image_cropped = pil_image.crop((left, top, left + tw, top + th))
        pil_depth_gt_cropped = pil_depth_gt.crop((left, top, left + tw, top + th)) if pil_depth_gt else None
        return pil_image_cropped, pil_depth_gt_cropped

    def _apply_nyu_crop(self, pil_image, pil_depth_gt=None):
        crop_box = (45, 41, 608, 472)
        pil_image_cropped = pil_image.crop(crop_box)
        pil_depth_gt_cropped = pil_depth_gt.crop(crop_box) if pil_depth_gt else None
        return pil_image_cropped, pil_depth_gt_cropped
        
    def _common_pil_to_numpy_and_scale_depth(self, pil_image, pil_depth_gt):
        image_np = np.asarray(pil_image, dtype=np.float32) / 255.0
        depth_np = np.array([])
        if pil_depth_gt:
            depth_tmp = np.asarray(pil_depth_gt, dtype=np.float32)
            if self.args.dataset.lower() == 'nyu': depth_tmp /= 1000.0
            elif self.args.dataset.lower() == 'kitti': depth_tmp /= 256.0
            if depth_tmp.ndim == 2: depth_tmp = np.expand_dims(depth_tmp, axis=2)
            depth_np = depth_tmp
        return image_np, depth_np

    def _preprocess_train(self, pil_image, pil_depth_gt):
        if getattr(self.args, 'do_kb_crop', False) and self.args.dataset.lower() == 'kitti':
            pil_image, pil_depth_gt = self._apply_kb_crop(pil_image, pil_depth_gt)
        if getattr(self.args, 'do_nyu_crop', True) and self.args.dataset.lower() == 'nyu':
            pil_image, pil_depth_gt = self._apply_nyu_crop(pil_image, pil_depth_gt)

        if getattr(self.args, 'do_random_rotate', False):
            angle = (random.random() - 0.5) * 2 * self.args.degree
            pil_image = self.rotate_image(pil_image, angle, flag=Image.BICUBIC)
            pil_depth_gt = self.rotate_image(pil_depth_gt, angle, flag=Image.NEAREST)

        image_np, depth_np = self._common_pil_to_numpy_and_scale_depth(pil_image, pil_depth_gt)
        
        image_np, depth_np = self.random_crop(image_np, depth_np, self.target_height, self.target_width)
        image_np, depth_np = self.train_preprocess_numpy(image_np, depth_np)
        return image_np, depth_np

    def _preprocess_eval_test(self, pil_image, pil_depth_gt):
        if getattr(self.args, 'do_kb_crop', False) and self.args.dataset.lower() == 'kitti':
            pil_image, pil_depth_gt = self._apply_kb_crop(pil_image, pil_depth_gt)
        if getattr(self.args, 'do_nyu_crop_eval', False) and self.args.dataset.lower() == 'nyu':
             pil_image, pil_depth_gt = self._apply_nyu_crop(pil_image, pil_depth_gt)

        image_np, depth_np = self._common_pil_to_numpy_and_scale_depth(pil_image, pil_depth_gt)
        has_valid_depth = True if pil_depth_gt and depth_np.size > 0 else False

        if image_np.shape[0] != self.target_height or image_np.shape[1] != self.target_width:
            pil_img_to_resize = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_img_resized = pil_img_to_resize.resize((self.target_width, self.target_height), Image.BICUBIC)
            image_np = np.asarray(pil_img_resized, dtype=np.float32) / 255.0
            if has_valid_depth:
                pil_depth_to_resize = Image.fromarray(depth_np.squeeze())
                pil_depth_resized = pil_depth_to_resize.resize((self.target_width, self.target_height), Image.NEAREST)
                depth_np = np.expand_dims(np.asarray(pil_depth_resized, dtype=np.float32), axis=2)
        
        return image_np, depth_np, has_valid_depth

    def rotate_image(self, image, angle, flag=Image.BICUBIC):
        fill = 0 if image.mode in ['F', 'I', 'I;16', 'L'] else (0,0,0)
        return image.rotate(angle, resample=flag, fillcolor=fill)

    def random_crop(self, img_np, depth_np, target_height, target_width):
        h_img, w_img = img_np.shape[:2]
        if h_img < target_height or w_img < target_width:
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
            pil_depth = Image.fromarray(depth_np.squeeze())
            pil_img = pil_img.resize((target_width, target_height), Image.BICUBIC)
            pil_depth = pil_depth.resize((target_width, target_height), Image.NEAREST)
            img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
            depth_np = np.expand_dims(np.asarray(pil_depth, dtype=np.float32), axis=2)
            return img_np, depth_np
        start_x = random.randint(0, w_img - target_width)
        start_y = random.randint(0, h_img - target_height)
        return (img_np[start_y:start_y + target_height, start_x:start_x + target_width, :],
                depth_np[start_y:start_y + target_height, start_x:start_x + target_width, :])

    def train_preprocess_numpy(self, image_np, depth_gt_np):
        if random.random() > 0.5:
            image_np = np.ascontiguousarray(image_np[:, ::-1, :])
            depth_gt_np = np.ascontiguousarray(depth_gt_np[:, ::-1, :])
        if random.random() > 0.5:
            image_np = self.augment_image_numpy(image_np)
        return image_np, depth_gt_np

    def augment_image_numpy(self, image_np):
        gamma = random.uniform(0.9, 1.1)
        image_aug_np = image_np ** gamma
        brightness = random.uniform(0.9, 1.1)
        image_aug_np *= brightness
        colors = np.random.uniform(0.9, 1.1, size=3)
        for c in range(3): image_aug_np[:, :, c] *= colors[c]
        image_aug_np = np.clip(image_aug_np, 0, 1)
        return image_aug_np

    def __len__(self):
        return len(self.filenames)

class ToTensor(object):
    """
    Numpy 배열 샘플을 PyTorch 텐서로 변환하는 클래스.
    """
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                              std=[0.26862954, 0.26130258, 0.27577711])

    def __call__(self, sample):
        image_clip_np, image_orig_np, focal = sample['image_clip'], sample['image_orig'], sample['focal']
        
        image_clip_tensor = torch.from_numpy(image_clip_np.transpose((2, 0, 1))).float()
        image_clip_tensor = self.normalize(image_clip_tensor)
        
        image_orig_tensor = torch.from_numpy(image_orig_np.transpose((2, 0, 1))).float()
        
        output = {'image_clip': image_clip_tensor, 'image_orig': image_orig_tensor, 'focal': torch.tensor(focal).float()}

        if self.mode == 'train' or self.mode == 'online_eval':
            depth_np = sample.get('depth')
            
            if self.mode == 'online_eval':
                output['has_valid_depth'] = sample.get('has_valid_depth', False)
                output['image_path'] = sample.get('image_path', "")
                output['depth_path'] = sample.get('depth_path', "")
            
            if sample.get('has_valid_depth', True) and depth_np is not None and depth_np.size > 0:
                if depth_np.ndim == 2: depth_np = np.expand_dims(depth_np, axis=2)
                output['depth'] = torch.from_numpy(depth_np.transpose((2, 0, 1))).float()
            elif self.mode == 'online_eval':
                h, w = image_orig_np.shape[:2]
                output['depth'] = torch.zeros((1, h, w), dtype=torch.float32)
                
        return output