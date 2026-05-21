"""
Local copy of P3Dataset modified for AI4SmallFarms.
Key changes:
- Patches with total vertex count > max_num_vertices are filtered out at init.
- Coordinates are defensively clipped after augmentation to stay in pixel space.
- Safe tensor conversion works with both NumPy 1.x and 2.x.
"""
import os
import cv2
import torch
import numpy as np
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from sklearn.preprocessing import MinMaxScaler
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from pixelspointspolygons.misc import make_logger, suppress_stdout


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def _safe_to_tensor(arr):
    """Convert a NumPy array to a PyTorch FloatTensor, safe with NumPy 2.x."""
    if isinstance(arr, torch.Tensor):
        return arr.float()
    return torch.FloatTensor(np.ascontiguousarray(arr).tolist())


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


# --------------------------------------------------------------------
# Main dataset class
# --------------------------------------------------------------------
class P3Dataset(Dataset):
    def __init__(self, cfg, split, transform=None, **kwargs):
        super().__init__()
        self.logger = make_logger(f'{split}Dataset', cfg.run_type.logging)
        self.cfg = cfg
        self.split = split

        self.dataset_dir = cfg.experiment.dataset.in_path
        if not os.path.isdir(self.dataset_dir):
            raise NotADirectoryError(f"Dataset directory {self.dataset_dir} does not exist")

        # FFL requires a separate annotation file
        if cfg.experiment.model.name == "ffl":
            self.ann_file = cfg.experiment.dataset.annotations[split].replace(
                "annotations_", "annotations_ffl_")
            self.stats_filepath = cfg.experiment.dataset.ffl_stats[split]
            if not os.path.isfile(self.stats_filepath):
                raise FileExistsError(self.stats_filepath)
            self.stats = torch.load(self.stats_filepath)
        else:
            self.ann_file = cfg.experiment.dataset.annotations[split]
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError(self.ann_file)

        with suppress_stdout():
            self.coco = COCO(self.ann_file)
        images_id = self.coco.getImgIds()

        # Filter out patches that contain more vertices than the tokenizer allows.
        # Without this, they would be silently truncated → incomplete polygon sequences.
        max_v = cfg.experiment.model.tokenizer.max_num_vertices
        filtered_ids = []
        for img_id in images_id:
            total_verts = sum(
                len(seg) // 2 - 1               # closed polygon: first == last
                for ann in self.coco.imgToAnns.get(img_id, [])
                for seg in ann["segmentation"]
            )
            if total_verts <= max_v:
                filtered_ids.append(img_id)
        n_removed = len(images_id) - len(filtered_ids)
        self.tile_ids = filtered_ids
        self.num_samples = len(self.tile_ids)

        self.logger.info(
            f"Loaded {len(self.coco.anns.items())} annotations from "
            f"{len(self.coco.imgs.items())} images from {self.ann_file} "
            f"(removed {n_removed} patches exceeding {max_v} vertices)"
        )

        self.use_lidar = cfg.experiment.encoder.use_lidar
        self.use_images = cfg.experiment.encoder.use_images
        self.transform = transform
        self.model_type = cfg.experiment.model.name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    # I/O stubs (LiDAR disabled – returns None)
    # ------------------------------------------------------------------
    def load_lidar_points(self, lidar_file_name, img_info):
        # LiDAR loading is disabled for now; returns None.
        return None

    def get_image_file(self, coco_info):
        filename = os.path.join(self.dataset_dir, coco_info['image_path'])
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        return filename

    def get_lidar_file(self, coco_info):
        return None

    # ------------------------------------------------------------------
    # Augmentation helpers (LiDAR & FFL crossfield angle)
    # ------------------------------------------------------------------
    def apply_d4_augmentations_to_lidar(self, augmentation_replay, lidar):
        """Disabled LiDAR D4 augmentation."""
        return None

    def apply_augmentations_to_ffl_crossfield_angle(
        self, crossfield_angle_mask, augmentation_replay=None, group_element=None
    ):
        """Rotate crossfield angle according to D4 group element (train only)."""
        if self.split == 'val':
            return crossfield_angle_mask

        if augmentation_replay is not None:
            d4_transform = augmentation_replay["transforms"][0]
            if d4_transform["__class_fullname__"] != "D4" or not d4_transform['applied']:
                return crossfield_angle_mask

        if group_element is None:
            group_element = d4_transform['params']['group_element']

        # Adjust angle based on D4 action
        if group_element == 'e':
            pass
        elif group_element == 'r90':
            crossfield_angle_mask = (crossfield_angle_mask + np.pi / 2) % np.pi
        elif group_element == 'r180':
            crossfield_angle_mask = (crossfield_angle_mask + np.pi) % np.pi
        elif group_element == 'r270':
            crossfield_angle_mask = (crossfield_angle_mask + 3 * np.pi / 2) % np.pi
        elif group_element == 'v':
            crossfield_angle_mask = (np.pi - crossfield_angle_mask) % np.pi
        elif group_element == 'hvt':
            crossfield_angle_mask = (3 * np.pi / 2 - crossfield_angle_mask) % np.pi
        elif group_element == 'h':
            crossfield_angle_mask = (-crossfield_angle_mask) % np.pi
        elif group_element == 't':
            crossfield_angle_mask = (np.pi / 2 - crossfield_angle_mask) % np.pi
        else:
            raise ValueError(f"Unknown group element {group_element}")
        return crossfield_angle_mask

    # ------------------------------------------------------------------
    # __getitem__ dispatcher
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        if self.model_type == 'hisup':
            return self.__getitem__hisup(idx)
        elif self.model_type == 'pix2poly':
            return self.__getitem__pix2poly(idx)
        elif self.model_type == 'ffl':
            return self.__getitem__ffl(idx)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented.")

    # ------------------------------------------------------------------
    # FFL-specific item loading (not used in current experiments)
    # ------------------------------------------------------------------
    def __getitem__ffl(self, index):
        img_id = self.tile_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        if self.use_images:
            img_file = self.get_image_file(img_info)
            with rasterio.open(img_file) as src:
                image = src.read([1, 2, 3])  # (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # HWC
        else:
            image = np.zeros((img_info['width'], img_info['height'], 1), dtype=np.uint8)

        # LiDAR not used
        lidar = None

        # Load pre-computed FFL data
        ffl_pt_file = os.path.join(self.dataset_dir, img_info["ffl_pt_path"])
        if not os.path.isfile(ffl_pt_file):
            raise FileExistsError(ffl_pt_file)
        ffl_data = torch.load(ffl_pt_file, weights_only=False)
        ffl_data["image_id"] = torch.IntTensor([img_id])

        if self.transform is not None:
            masks = [
                ffl_data["gt_polygons_image"][:, :, 0],   # interior
                ffl_data["gt_polygons_image"][:, :, 1],   # edges
                ffl_data["gt_polygons_image"][:, :, 2],   # vertices
                ffl_data["distances"],
                ffl_data["sizes"],
                ffl_data["gt_crossfield_angle"],
            ]
            augmentations = self.transform(image=image, masks=masks)

            if self.use_lidar:
                lidar = self.apply_d4_augmentations_to_lidar(augmentations["replay"], lidar)
            ffl_data["image"] = augmentations['image']

            # Reconstruct polygon image
            gt_polygon_image = []
            for i in range(3):
                gt_polygon_image.append(augmentations['masks'][i])
            ffl_data["gt_polygons_image"] = (
                torch.stack(gt_polygon_image, axis=-1).permute(2, 0, 1) / 255.0
            )
            ffl_data["gt_polygons_image"] = torch.clamp(
                ffl_data["gt_polygons_image"], 0, 1
            ).float()

            ffl_data["distances"] = augmentations['masks'][3][None, ...]
            ffl_data["sizes"] = augmentations['masks'][4][None, ...]

            # Adjust crossfield angle: stored normals → tangents, then apply D4
            ffl_data["gt_crossfield_angle"] = augmentations['masks'][5] * np.pi / 255.0
            ffl_data["gt_crossfield_angle"] = (
                ffl_data["gt_crossfield_angle"] + np.pi / 2
            ) % np.pi
            ffl_data["gt_crossfield_angle"] = self.apply_augmentations_to_ffl_crossfield_angle(
                ffl_data["gt_crossfield_angle"],
                augmentation_replay=augmentations["replay"]
            )[None, ...]

        ffl_data["class_freq"] = torch.from_numpy(self.stats["class_freq"])
        return ffl_data

    # ------------------------------------------------------------------
    # Permutation matrix helpers
    # ------------------------------------------------------------------
    def shuffle_perm_matrix_by_indices(self, old_perm: torch.Tensor, shuffle_idxs: np.ndarray):
        Nv = old_perm.shape[0]
        padd_idxs = np.arange(len(shuffle_idxs), Nv)
        shuffle_idxs = np.concatenate([shuffle_idxs, padd_idxs], axis=0)
        transform_arr = torch.zeros_like(old_perm)
        for i in range(len(shuffle_idxs)):
            transform_arr[i, shuffle_idxs[i]] = 1.
        new_perm = torch.mm(torch.mm(transform_arr, old_perm), transform_arr.T)
        return new_perm

    def add_vertex_valence_to_seq(self, coords_seq: list):
        arr = np.array(coords_seq)[1:-1].reshape(-1, 2)
        uniq, counts = np.unique(arr, axis=0, return_counts=True)
        _, idx = np.unique(arr, axis=0, return_inverse=True)
        counts_col = counts[idx].reshape(-1, 1)
        result = np.hstack([arr, counts_col])
        result = result.flatten().tolist()
        return [coords_seq[0]] + result + [coords_seq[-1]]

    # ------------------------------------------------------------------
    # Main pix2poly item loading (used during training)
    # ------------------------------------------------------------------
    def __getitem__pix2poly(self, index: int):
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError("Tokenizer not set. Please pass a tokenizer to the dataset.")

        img_id = self.tile_ids[index]
        img_info = self.coco.imgs[img_id]

        # Load image
        if self.use_images:
            img_file = self.get_image_file(img_info)
            with rasterio.open(img_file) as src:
                image = src.read([1, 2, 3])  # (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # HWC
        else:
            image = np.zeros((img_info['width'], img_info['height'], 1), dtype=np.uint8)

        # LiDAR not used
        lidar = None

        corner_coords = []
        max_verts = self.cfg.experiment.model.tokenizer.max_num_vertices
        perm_matrix = np.zeros((max_verts, max_verts), dtype=np.float32)
        annotations = self.coco.imgToAnns[img_id]

        if self.cfg.experiment.model.shuffle_polygons:
            np.random.shuffle(annotations)

        # Extract polygon vertices (pixel space, clipped to image bounds)
        for ann in annotations:
            for poly in ann['segmentation']:
                poly = np.array(poly).reshape(-1, 2)  # (x, y)
                poly[:, 0] = np.clip(poly[:, 0], 0, img_info['width'] - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, img_info['height'] - 1)
                assert (poly[0] == poly[-1]).all(), \
                    "COCO polygons must be closed (first == last)."
                points = poly[:-1]  # remove closing point
                corner_coords.extend(points.tolist())

        # Convert to (row, col) = (y, x) for albumentations (format='yx')
        corner_coords = np.flip(np.round(corner_coords, 0), axis=-1).astype(np.int32)

        # Build ground-truth permutation matrix
        v_count = 0
        for ann in annotations:
            for poly in ann['segmentation']:
                poly = np.array(poly).reshape(-1, 2)
                assert (poly[0] == poly[-1]).all()
                points = poly[:-1]
                for i in range(len(points)):
                    j = (i + 1) % len(points)
                    if v_count + i >= max_verts or v_count + j >= max_verts:
                        break
                    perm_matrix[v_count + i, v_count + j] = 1.
                v_count += len(points)

        # Fill diagonal for remaining vertices (padding)
        for i in range(v_count, max_verts):
            perm_matrix[i, i] = 1.

        # Fix open contours: if a vertex has no edge, set self-loop
        for i in range(max_verts):
            if np.sum(perm_matrix[i, :]) == 0 or np.sum(perm_matrix[:, i]) == 0:
                perm_matrix[i, i] = 1.

        perm_matrix = _safe_to_tensor(perm_matrix)

        # Truncate if too many vertices
        if len(corner_coords) > max_verts:
            corner_coords = corner_coords[:max_verts]

        # Apply augmentations (D4 + Normalize, defined in build_datasets)
        if self.transform is not None:
            augmentations = self.transform(image=image, keypoints=corner_coords.tolist())
            if self.use_lidar:
                lidar = self.apply_d4_augmentations_to_lidar(augmentations["replay"], lidar)
            image = augmentations['image']
            corner_coords = np.array(augmentations['keypoints'])

            # Defensive clip: keypoints may slightly overflow after rotation.
            # Without this, quantize() can produce token 35 (= vocab_size) → CUDA crash.
            if len(corner_coords) > 0:
                h, w = img_info['height'], img_info['width']
                corner_coords[:, 0] = np.clip(corner_coords[:, 0], 0, h - 1)  # row
                corner_coords[:, 1] = np.clip(corner_coords[:, 1], 0, w - 1)  # col

        # Tokenise
        coords_seqs, rand_idxs = self.tokenizer(
            corner_coords,
            shuffle=self.cfg.experiment.model.tokenizer.shuffle_tokens
        )
        coords_seqs = torch.LongTensor(coords_seqs)
        if self.cfg.experiment.model.tokenizer.shuffle_tokens:
            perm_matrix = self.shuffle_perm_matrix_by_indices(perm_matrix, rand_idxs)

        # Ensure image is a float tensor (C, H, W)
        if not isinstance(image, torch.Tensor):
            image = _safe_to_tensor(image)
            if image.dim() == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
        elif image.dtype != torch.float32:
            image = image.float()

        return image, lidar, coords_seqs, perm_matrix, torch.tensor([img_info['id']])

    # ------------------------------------------------------------------
    # pix2poly variant (without the defensive clip – kept for reference)
    # ------------------------------------------------------------------
    def __getitem__pix2poly_polygons(self, index: int):
        # Identical to __getitem__pix2poly except the defensive clip after
        # augmentation is omitted (used in evaluation / validation path where
        # we trust the augmentations less).
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError("Tokenizer not set.")

        img_id = self.tile_ids[index]
        img_info = self.coco.imgs[img_id]

        if self.use_images:
            img_file = self.get_image_file(img_info)
            with rasterio.open(img_file) as src:
                image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))
        else:
            image = np.zeros((img_info['width'], img_info['height'], 1), dtype=np.uint8)

        lidar = None
        corner_coords = []
        max_verts = self.cfg.experiment.model.tokenizer.max_num_vertices
        perm_matrix = np.zeros((max_verts, max_verts), dtype=np.float32)
        annotations = self.coco.imgToAnns[img_id]

        if self.cfg.experiment.model.shuffle_polygons:
            np.random.shuffle(annotations)

        for ann in annotations:
            for poly in ann['segmentation']:
                poly = np.array(poly).reshape(-1, 2)
                poly[:, 0] = np.clip(poly[:, 0], 0, img_info['width'] - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, img_info['height'] - 1)
                assert (poly[0] == poly[-1]).all()
                points = poly[:-1]
                corner_coords.extend(points.tolist())

        corner_coords = np.flip(np.round(corner_coords, 0), axis=-1).astype(np.int32)

        v_count = 0
        for ann in annotations:
            for poly in ann['segmentation']:
                poly = np.array(poly).reshape(-1, 2)
                assert (poly[0] == poly[-1]).all()
                points = poly[:-1]
                for i in range(len(points)):
                    j = (i + 1) % len(points)
                    if v_count + i >= max_verts or v_count + j >= max_verts:
                        break
                    perm_matrix[v_count + i, v_count + j] = 1.
                v_count += len(points)

        for i in range(v_count, max_verts):
            perm_matrix[i, i] = 1.
        for i in range(max_verts):
            if np.sum(perm_matrix[i, :]) == 0 or np.sum(perm_matrix[:, i]) == 0:
                perm_matrix[i, i] = 1.

        perm_matrix = _safe_to_tensor(perm_matrix)

        if len(corner_coords) > max_verts:
            corner_coords = corner_coords[:max_verts]

        if self.transform is not None:
            augmentations = self.transform(image=image, keypoints=corner_coords.tolist())
            if self.use_lidar:
                lidar = self.apply_d4_augmentations_to_lidar(augmentations["replay"], lidar)
            image = augmentations['image']
            corner_coords = np.array(augmentations['keypoints'])

        coords_seqs, rand_idxs = self.tokenizer(
            corner_coords,
            shuffle=self.cfg.experiment.model.tokenizer.shuffle_tokens
        )
        coords_seqs = torch.LongTensor(coords_seqs)
        if self.cfg.experiment.model.tokenizer.shuffle_tokens:
            perm_matrix = self.shuffle_perm_matrix_by_indices(perm_matrix, rand_idxs)

        return image, lidar, coords_seqs, perm_matrix, torch.tensor([img_info['id']])

    # ------------------------------------------------------------------
    # HiSup-specific (not used)
    # ------------------------------------------------------------------
    def __getitem__hisup(self, index):
        img_id = self.tile_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        annotations = self.coco.loadAnns(ann_ids)

        if self.use_images:
            img_file = self.get_image_file(img_info)
            with rasterio.open(img_file) as src:
                image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))
        else:
            image = np.zeros((img_info['width'], img_info['height'], 1), dtype=np.uint8)

        lidar = None
        corner_coords = []
        corner_poly_ids = [0]
        mask = np.zeros([img_info['width'], img_info['height']])

        for i, ann in enumerate(annotations):
            mask += self.coco.annToMask(ann)
            seg = ann['segmentation']
            if len(seg) > 1:
                raise ValueError("Multipolygon not supported.")
            points = np.array(seg[0]).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, img_info['width'] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, img_info['height'] - 1)
            points = points[:-1]
            corner_poly_ids.append(len(points) + len(corner_coords))
            corner_coords.extend(points.tolist())

        mask = np.clip(mask / 255.0 if mask.max() == 255 else mask, 0, 1)

        if self.transform is not None:
            corner_coords = np.flip(corner_coords, axis=-1)
            augmentations = self.transform(image=image, masks=[mask], keypoints=corner_coords)
            if self.use_lidar:
                lidar = self.apply_d4_augmentations_to_lidar(augmentations["replay"], lidar)
            image = augmentations['image']
            corner_coords = np.flip(augmentations['keypoints'], axis=-1)

        ann = self.make_hisup_annotations(corner_coords, corner_poly_ids,
                                          img_info['height'], img_info['width'])
        ann["mask"] = augmentations['masks'][0]

        if (self.cfg.experiment.model.decoder.in_feature_width != img_info['width'] or
            self.cfg.experiment.model.decoder.in_feature_height != img_info['height']):
            self.resize_hisup_annotations(ann)
        else:
            ann['mask_ori'] = ann['mask'].clone()

        for k, v in ann.items():
            if isinstance(v, np.ndarray):
                ann[k] = torch.from_numpy(v)

        return image, lidar, ann, torch.tensor([img_info['id']])

    def resize_hisup_annotations(self, ann):
        sx = self.cfg.experiment.model.decoder.in_feature_width / ann['width']
        sy = self.cfg.experiment.model.decoder.in_feature_height / ann['height']
        ann['junc_ori'] = ann['junctions'].copy()
        ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0,
                                         self.cfg.experiment.model.decoder.in_feature_width - 1e-4)
        ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0,
                                         self.cfg.experiment.model.decoder.in_feature_height - 1e-4)
        ann['width'] = self.cfg.experiment.model.decoder.in_feature_width
        ann['height'] = self.cfg.experiment.model.decoder.in_feature_height
        ann['mask_ori'] = ann['mask'].clone()
        ann['mask'] = cv2.resize(
            np.array(ann['mask']).astype(np.uint8),
            (int(ann['width']), int(ann['height']))
        )

    def make_hisup_annotations(self, corner_coords, corner_poly_ids, height, width):
        ann = {
            'junctions': [], 'juncs_index': [], 'juncs_tag': [],
            'edges_positive': [], 'bbox': [], 'width': width, 'height': height
        }
        pid = 0
        instance_id = 0
        for i in range(len(corner_poly_ids) - 1):
            juncs, tags = [], []
            points = corner_coords[corner_poly_ids[i]: corner_poly_ids[i + 1]]
            junc_tags = np.ones(len(points))
            poly = Polygon(points)
            if poly.area > 0:
                convex_point = np.array(poly.convex_hull.exterior.coords)[:-1]
                convex_index = [(p == convex_point).all(1).any() for p in points]
                juncs.extend(points.tolist())
                junc_tags = np.array([2 if c else 1 for c in convex_index])  # 2 = convex
                tags.extend(junc_tags.tolist())
                ann['bbox'].append(list(poly.bounds))

                idxs = np.arange(len(juncs))
                edges = np.stack((idxs, np.roll(idxs, 1))).T + pid
                ann['juncs_index'].extend([instance_id] * len(juncs))
                ann['junctions'].extend(juncs)
                ann['juncs_tag'].extend(tags)
                ann['edges_positive'].extend(edges.tolist())
                instance_id += 1
                pid += len(juncs)

        # Fallback if empty
        if len(ann['junctions']) == 0:
            ann.update({
                'junctions': np.array([[0, 0]], dtype=np.float32),
                'bbox': np.array([[0, 0, 0, 0]], dtype=np.float32),
                'juncs_tag': np.array([0], dtype=np.longlong),
                'juncs_index': np.array([0], dtype=np.longlong),
            })
        else:
            for key, t in zip(['junctions', 'edges_positive', 'juncs_tag', 'juncs_index', 'bbox'],
                              [np.float32, np.longlong, np.longlong, np.longlong, np.float32]):
                ann[key] = np.array(ann[key], dtype=t)
        return ann


# --------------------------------------------------------------------
# Dataset subclasses (just bind the split)
# --------------------------------------------------------------------
class TestDataset(P3Dataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, 'test', **kwargs)

class ValDataset(P3Dataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, 'val', **kwargs)

class TrainDataset(P3Dataset):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, 'train', **kwargs)