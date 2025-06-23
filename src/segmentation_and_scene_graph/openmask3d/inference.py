import os
import numpy as np
import open3d as o3d
import torch
import clip
import copy
import pdb
import matplotlib.pyplot as plt
import csv
from constants import *
from sklearn.cluster import DBSCAN


class QuerySimilarityComputation():
    def __init__(self,):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.clip_model, _ = clip.load('ViT-L/14@336px', self.device)

    def get_query_embedding(self, text_query):
        text_input_processed = clip.tokenize(text_query).to(self.device)
        with torch.no_grad():
            sentence_embedding = self.clip_model.encode_text(text_input_processed)

        sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
        return sentence_embedding_normalized.squeeze().numpy()
 
    def compute_similarity_scores(self, mask_features, text_query):
        text_emb = self.get_query_embedding(text_query)

        scores = np.zeros(len(mask_features))
        for mask_idx, mask_emb in enumerate(mask_features):
            mask_norm = np.linalg.norm(mask_emb)
            if mask_norm < 0.001:
                continue
            normalized_emb = (mask_emb/mask_norm)
            scores[mask_idx] = normalized_emb@text_emb

        return scores
    
    def get_per_point_colors_for_similarity(self, 
                                            per_mask_scores, 
                                            masks, 
                                            top_k_masks,
                                            normalize_based_on_current_min_max=False, 
                                            normalize_min_bound=0.16, #only used for visualization if normalize_based_on_current_min_max is False
                                            normalize_max_bound=0.26, #only used for visualization if normalize_based_on_current_min_max is False
                                            background_color=(0.77, 0.77, 0.77)
                                        ):
        # get colors based on the openmask3d per mask scores
        non_zero_points = per_mask_scores!=0
        openmask3d_per_mask_scores_rescaled = np.zeros_like(per_mask_scores)
        pms = per_mask_scores[non_zero_points]

        # in order to be able to visualize the score differences better, we can use a normalization scheme
        if normalize_based_on_current_min_max: # if true, normalize the scores based on the min. and max. scores for this scene
            openmask3d_per_mask_scores_rescaled[non_zero_points] = (pms-pms.min()) / (pms.max() - pms.min())
        else: # if false, normalize the scores based on a pre-defined color scheme with min and max clipping bounds, normalize_min_bound and normalize_max_bound.
            new_scores = np.zeros_like(openmask3d_per_mask_scores_rescaled)
            new_indices = np.zeros_like(non_zero_points)
            new_indices[non_zero_points] += pms>normalize_min_bound
            new_scores[new_indices] = ((pms[pms>normalize_min_bound]-normalize_min_bound)/(normalize_max_bound-normalize_min_bound))
            openmask3d_per_mask_scores_rescaled = new_scores

        new_colors = np.ones((masks.shape[1], 3))*0 + background_color

        scores = copy.deepcopy(openmask3d_per_mask_scores_rescaled)
        top_k_indices = np.argsort(scores)[-top_k_masks:]

        # Apply colors from colormap
        for i, mask_idx in enumerate(top_k_indices):
            mask = masks[mask_idx, :]
            color = plt.cm.jet(scores[mask_idx])[:3]
            new_colors[mask > 0.5, :] = color

        return new_colors


def clean_small_object_with_clustering(points, mask_indices, min_points_threshold=5000, eps=0.05, min_samples=10):
    """
    Apply clustering to small objects to remove outlier points.
    
    Args:
        points: All scene points (numpy array)
        mask_indices: Indices of points belonging to the current mask
        min_points_threshold: Threshold below which clustering is applied
        eps: DBSCAN neighborhood distance parameter
        min_samples: Minimum samples in DBSCAN neighborhood
    
    Returns:
        cleaned_mask_indices: Indices of points after removing outliers
        outlier_indices: Indices of outlier points that should be colored white
    """
    if len(mask_indices) >= min_points_threshold:
        # Object is large enough, no cleaning needed
        return mask_indices, []
    
    # Get the 3D coordinates of the masked points
    object_points = np.asarray(points)[mask_indices]
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(object_points)
    labels = clustering.labels_
    
    # Find the largest cluster (most points)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    
    if len(unique_labels) == 0:
        # No valid clusters found, keep all points
        return mask_indices, []
    
    # Get the label of the largest cluster
    largest_cluster_label = unique_labels[np.argmax(counts)]
    
    # Find points belonging to the largest cluster
    main_cluster_mask = labels == largest_cluster_label
    outlier_mask = labels == -1  # DBSCAN assigns -1 to noise/outliers
    other_clusters_mask = (labels >= 0) & (labels != largest_cluster_label)
    
    # Combine outliers and small clusters as points to be colored white
    points_to_remove_mask = outlier_mask | other_clusters_mask
    
    # Get the actual indices
    cleaned_mask_indices = mask_indices[main_cluster_mask]
    outlier_indices = mask_indices[points_to_remove_mask]
    
    print(f"  Clustering applied: {len(mask_indices)} -> {len(cleaned_mask_indices)} points (removed {len(outlier_indices)} outliers)")
    
    return cleaned_mask_indices, outlier_indices


def main():
    # --------------------------------
    # Set the paths
    # --------------------------------
    path_scene_pcd = "/home/cvg-robotics/project9_ws/SpotMap/data/data_om3/scene.ply"
    path_pred_masks = "/home/cvg-robotics/project9_ws/SpotMap/data/data_om3/openmask3d/scene_MASKS.pt"
    path_openmask3d_features = "/home/cvg-robotics/project9_ws/SpotMap/data/data_om3/openmask3d/clip_features.npy"
    
    # --------------------------------
    # Load data
    # --------------------------------
    # load the scene pcd
    scene_pcd = o3d.io.read_point_cloud(path_scene_pcd)
    
    # load the predicted masks
    pred_masks = np.asarray(torch.load(path_pred_masks, weights_only=False)).T

    # load the openmask3d features
    openmask3d_features = np.load(path_openmask3d_features) # (num_instances, 768)

    # initialize the query similarity computer
    query_similarity_computer = QuerySimilarityComputation()

    BOTTLE          = "white plastic bottle"   
    FLOWER_POT      = "rustic metal flower pot with artificial flowers with 'FLOWERS & GARDEN text"   
    LAMPSHADE       = "beige cylindrical lampshade with raised rope-like patterns"
    ARMCHAIR        = "yellow armchair"    
    CUSHION         = "round, pleated velvet cushion"
    SIDE_TABLE      = "small round white side table"
    TABLE_LAMP      = "table lamp with a white spherical light and curved green panels"          
    CABINET         = "gray cabinet with a QR code mounted on it"                  
    WATERING_CAN    = "green plastic watering can"                
    # --------------------------------
    # Set the query texts and colors
    # --------------------------------
    query_texts = [BOTTLE, FLOWER_POT, LAMPSHADE, ARMCHAIR, CUSHION, TABLE_LAMP, SIDE_TABLE, CABINET, WATERING_CAN]  
    hex_colors = ['#911eb4','#4363d8','#3cb44b','#ffe119','#42d4f4','#f58231','#ffd8b1','#000075','#aaffc3']  # Corresponding hex colors for the queries
    
    # Convert hex colors to RGB
    rgb_colors = []
    for hex_color in hex_colors:
        rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (1, 3, 5))
        rgb_colors.append(rgb)

    # --------------------------------
    # Process each query and find best masks
    # --------------------------------
    # Initialize colors for all points (white background)
    num_points = len(scene_pcd.points)
    per_point_colors = np.ones((num_points, 3))  # white background
    
    # Store mapping for CSV
    color_label_mapping = []
    
    used_masks = set()  # Keep track of used masks to avoid conflicts
    
    for query_idx, query_text in enumerate(query_texts):
        # Get the similarity scores for this query
        per_mask_query_sim_scores = query_similarity_computer.compute_similarity_scores(openmask3d_features, query_text)
        
        # Find the best mask for this query that hasn't been used yet
        sorted_indices = np.argsort(per_mask_query_sim_scores)[::-1]  # descending order
        
        best_mask_idx = None
        best_score = 0
        
        for mask_idx in sorted_indices:
            if mask_idx not in used_masks and per_mask_query_sim_scores[mask_idx] > 0.1:  # minimum threshold
                best_mask_idx = mask_idx
                best_score = per_mask_query_sim_scores[mask_idx]
                break
        
        if best_mask_idx is not None:
            used_masks.add(best_mask_idx)
            
            # Get the mask and color the points
            mask = pred_masks[best_mask_idx, :]
            mask_indices = np.where(mask > 0.5)[0]
            
            if len(mask_indices) > 0:
                # Apply clustering to clean small objects
                cleaned_mask_indices, outlier_indices = clean_small_object_with_clustering(
                    scene_pcd.points, mask_indices, min_points_threshold=5000
                )
                
                # Color the main cluster with the object color
                per_point_colors[cleaned_mask_indices] = rgb_colors[query_idx]
                
                # Color outliers white (background color)
                if len(outlier_indices) > 0:
                    per_point_colors[outlier_indices] = [1.0, 1.0, 1.0]  # white
                
                # Add to mapping (using cleaned data)
                color_label_mapping.append({
                    'hex_color': hex_colors[query_idx],
                    'label': query_text,
                    'similarity_score': best_score,
                    'mask_index': best_mask_idx,
                    'num_points': len(cleaned_mask_indices),
                    'original_points': len(mask_indices),
                    'outliers_removed': len(outlier_indices)
                })
                
                print(f"Query '{query_text}': Mask {best_mask_idx}, Score = {best_score:.3f}, Points = {len(cleaned_mask_indices)}/{len(mask_indices)}")

    # --------------------------------
    # Create and save colored point cloud
    # --------------------------------
    scene_pcd_labeled = o3d.geometry.PointCloud()
    scene_pcd_labeled.points = scene_pcd.points
    scene_pcd_labeled.colors = o3d.utility.Vector3dVector(per_point_colors)
    scene_pcd_labeled.estimate_normals()
    
    # Save the labeled point cloud
    o3d.io.write_point_cloud("/home/cvg-robotics/project9_ws/SpotMap/scene_graph/scene.ply", scene_pcd_labeled)
    print("Saved labeled point cloud as 'scene.ply'")
    
    # --------------------------------
    # Save CSV mapping
    # --------------------------------
    # Define output directory
    output_dir = "/home/cvg-robotics/project9_ws/SpotMap/scene_graph"
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the full path for the CSV file
    csv_output_path = os.path.join(output_dir, "color_label_mapping.csv")
    
    with open(csv_output_path, 'w', newline='') as csvfile:
        fieldnames = ['hex_color', 'label', 'similarity_score', 'mask_index', 'num_points', 'original_points', 'outliers_removed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for mapping in color_label_mapping:
            writer.writerow(mapping)
    
    print("Saved color-label mapping as 'color_label_mapping.csv'")
    
    # Optional: Display the result
    o3d.visualization.draw_geometries([scene_pcd_labeled])

if __name__ == "__main__":
    main()