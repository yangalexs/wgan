import open3d as o3d
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np

base_dir = 'data_segment'
pcd_files = glob(os.path.join(base_dir, '*.pcd'))

# Create rotation angles dictionary based on filenames
rotation_angles = {
    'bump1_n17d1.pcd': -17.1,
    'bump2_n18d8.pcd': -18.8,
    'bump3_n19.pcd': -19,
    'bump4_n17d3.pcd': -17.3,
    'bump5_n17d65.pcd': -17.65,
    # 'hole1_n17d4.pcd': -17.4,
    # 'hole2_n17d2.pcd': -17.2,
    # 'hole3_n17d2.pcd': -17.2,
    # 'hole4_n17d2.pcd': -17.2,
    # 'hole5_n17d2.pcd': -17.2,
    # 'regular1_n17d3.pcd': -17.3,
    # 'regular2_n17d8.pcd': -17.8
}


class RoadDataset(Dataset):
    def __init__(self, pcd_files, rotation_angles, target_rows=128, target_points=128):
        if not pcd_files:
            raise ValueError("No PCD files found in the specified directory")

        self.pcd_files = [f for f in pcd_files if os.path.basename(f) in rotation_angles]
        if not self.pcd_files:
            raise ValueError("No matching PCD files found with rotation angles")

        self.rotation_angles = rotation_angles
        self.target_rows = target_rows
        self.target_points = target_points
        print(f"Found {len(self.pcd_files)} valid PCD files")

    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return len(self.pcd_files)

    def normalize_rows(self, data_list):
        """
        Normalize number of rows to target_rows through down/up sampling
        data_list: List of arrays, each with shape [4, num_points]
        """
        current_rows = len(data_list)

        if current_rows > self.target_rows:
            # Down sampling: randomly select target_rows
            indices = np.random.choice(current_rows, self.target_rows, replace=False)
            indices.sort()  # Keep rows in order
            normalized_data = [data_list[i] for i in indices]

        else:
            # Up sampling: duplicate existing rows with small random perturbations
            normalized_data = data_list.copy()

            while len(normalized_data) < self.target_rows:
                # Randomly select a row to duplicate
                idx = np.random.randint(0, len(data_list))
                row_to_duplicate = data_list[idx].copy()

                # Add small random perturbations to x, y, z coordinates
                # Keep perturbations small (1% of the range) to maintain road structure
                for i in range(3):  # Only perturb x, y, z coordinates, not labels
                    coord_range = np.max(row_to_duplicate[i]) - np.min(row_to_duplicate[i])
                    perturbation = np.random.uniform(-0.01 * coord_range, 0.01 * coord_range, row_to_duplicate.shape[1])
                    row_to_duplicate[i] += perturbation

                normalized_data.append(row_to_duplicate)

            # Sort rows based on y-coordinate to maintain road structure
            row_positions = [np.mean(row[1]) for row in normalized_data]
            sorted_indices = np.argsort(row_positions)
            normalized_data = [normalized_data[i] for i in sorted_indices]

            # Trim to exact number if we added too many
            normalized_data = normalized_data[:self.target_rows]

        return normalized_data

    def normalize_row_points(self, row, target_points=None):
        """
        Normalize number of points in a row to target_points through down/up sampling
        row: array of shape [4, N] containing [x, y, z, label] data
        """
        if target_points is None:
            target_points = self.target_points

        current_points = row.shape[1]

        if current_points > target_points:
            indices = np.random.choice(current_points, target_points, replace=False)
            normalized_row = row[:, indices]

        else:
            normalized_row = np.zeros((4, target_points))
            normalized_row[:, :current_points] = row

            x_min, x_max = np.min(row[0, :]), np.max(row[0, :])
            y_min, y_max = np.min(row[1, :]), np.max(row[1, :])
            z_min, z_max = np.min(row[2, :]), np.max(row[2, :])

            points_to_add = target_points - current_points
            new_x = np.random.uniform(x_min, x_max, points_to_add)
            new_y = np.random.uniform(y_min, y_max, points_to_add)
            new_z = np.random.uniform(z_min, z_max, points_to_add)

            unique_labels, counts = np.unique(row[3, :], return_counts=True)
            majority_label = unique_labels[counts.argmax()]
            new_labels = np.full(points_to_add, majority_label)

            normalized_row[0, current_points:] = new_x
            normalized_row[1, current_points:] = new_y
            normalized_row[2, current_points:] = new_z
            normalized_row[3, current_points:] = new_labels

        return normalized_row

    def process_single_file(self, pcd_file):
        # Get rotation angle for this file
        file_name = os.path.basename(pcd_file)
        theta = np.radians(self.rotation_angles[file_name])

        # Read and preprocess point cloud
        pcd = o3d.t.io.read_point_cloud(pcd_file)
        xyz = pcd.point.positions.numpy()
        labels = pcd.point.label.numpy().reshape(-1).astype(np.int8)

        x = -xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Rotation
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

        y_shifted = -y + np.max(y)
        z_shifted = z
        rotated_coords = rotation_matrix @ np.vstack([y_shifted, z_shifted])

        x = x - np.min(x)
        y_rotated = rotated_coords[0, :] + np.max(y)
        z_rotated = rotated_coords[1, :] - rotated_coords[1, 1]
        y_rotated = y_rotated - y_rotated.min()

        # Sorting
        sorted_indices = np.lexsort((x, y_rotated))
        x_sorted = x[sorted_indices]
        y_sorted = y_rotated[sorted_indices]
        z_sorted = z_rotated[sorted_indices]
        labels_sorted = labels[sorted_indices]

        # Segment into rows
        data_list = []
        current_points = np.array([[x_sorted[0]],
                                   [y_sorted[0]],
                                   [z_sorted[0]],
                                   [labels_sorted[0]]])

        for i in range(1, len(x_sorted)):
            if abs(y_sorted[i] - current_points[1, 0]) > 0.023:
                # Sort current row based on x coordinates
                sort_indices = np.argsort(current_points[0, :])
                sorted_row = current_points[:, sort_indices]
                data_list.append(sorted_row)

                # Start new row
                current_points = np.array([[x_sorted[i]],
                                           [y_sorted[i]],
                                           [z_sorted[i]],
                                           [labels_sorted[i]]])
            else:
                # Add point to current row
                new_point = np.array([[x_sorted[i]],
                                      [y_sorted[i]],
                                      [z_sorted[i]],
                                      [labels_sorted[i]]])
                current_points = np.hstack([current_points, new_point])

        # Add last row after sorting
        sort_indices = np.argsort(current_points[0, :])
        sorted_row = current_points[:, sort_indices]
        data_list.append(sorted_row)
        # First normalize points in each row
        normalized_points_list = [self.normalize_row_points(row) for row in data_list]

        # Then normalize number of rows
        normalized_data_list = self.normalize_rows(normalized_points_list)

        return normalized_data_list

    def __getitem__(self, idx):
        pcd_file = self.pcd_files[idx]
        data_list = self.process_single_file(pcd_file)

        return {
            'data': data_list,  # List of exactly target_rows arrays, each with shape [4, target_points]
            'filename': os.path.basename(pcd_file),
            'num_rows': self.target_rows,
            'points_per_row': [row.shape[1] for row in data_list]
        }


# Modified analysis function to verify normalization
def analyze_sample(dataset, sample_idx):
    sample = dataset[sample_idx]
    print(f"\nAnalyzing file: {sample['filename']}")
    print(f"Number of rows: {sample['num_rows']}")
    print(f"Expected rows: {dataset.target_rows}")
    print(f"Expected points per row: {dataset.target_points}")
    print("\nPoints per row:")
    for i, points in enumerate(sample['points_per_row']):
        print(f"Row {i}: {points} points")
    print(f"\nTotal rows: {len(sample['data'])}")
    assert len(sample['data']) == dataset.target_rows, "Row count mismatch"
    assert all(row.shape[1] == dataset.target_points for row in sample['data']), "Point count mismatch"
    # Create dataset with 128 rows and 128 points per row

if __name__ == '__main__':

    dataset = RoadDataset(pcd_files, rotation_angles, target_rows=128, target_points=128)

    # Analyze a sample to verify normalization
    analyze_sample(dataset, 0)

    # 创建数据集
    dataset = RoadDataset(pcd_files, rotation_angles, target_rows=128, target_points=128)