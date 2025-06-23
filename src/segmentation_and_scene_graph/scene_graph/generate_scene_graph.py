import numpy as np
from scipy.spatial import KDTree, ConvexHull
import open3d as o3d
import os, pickle
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import datetime, time
import random
import pandas as pd
import warnings
from scipy.spatial.distance import cdist

def rgb_to_hex(rgb):
    """Convert RGB values (0-1 range) to hex string."""
    # Convert to 0-255 range and then to hex
    r, g, b = [int(c * 255) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def hex_to_rgb(hex_color):
    """Convert hex color string to RGB values (0-1 range)."""
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return np.array([r, g, b])

def find_unique_colors(colors, tolerance=0.01):
    """
    Find unique colors using a simple distance-based approach.
    
    Args:
        colors: Array of RGB colors (N, 3)
        tolerance: Color similarity tolerance
    
    Returns:
        unique_colors: List of unique colors
        color_labels: Label for each input color pointing to unique color index
    """
    unique_colors = []
    color_labels = np.zeros(len(colors), dtype=int)
    
    for i, color in enumerate(colors):
        found_match = False
        for j, unique_color in enumerate(unique_colors):
            if np.linalg.norm(color - unique_color) < tolerance:
                color_labels[i] = j
                found_match = True
                break
        
        if not found_match:
            unique_colors.append(color.copy())
            color_labels[i] = len(unique_colors) - 1
    
    return np.array(unique_colors), color_labels


class SpatialRelationship:
    """Class to represent spatial relationships between objects."""
    def __init__(self, from_id, to_id, relationship_type, confidence=1.0):
        self.from_id = from_id
        self.to_id = to_id
        self.relationship_type = relationship_type  # "on", "under", "in", "contains", "next_to"
        self.confidence = confidence
        
    def get_inverse(self):
        """Get the inverse relationship."""
        inverse_map = {
            "on": "under",
            "under": "on", 
            "in": "contains",
            "contains": "in",
            "next_to": "next_to"
        }
        return SpatialRelationship(
            self.to_id, 
            self.from_id, 
            inverse_map[self.relationship_type], 
            self.confidence
        )


class ObjectNode:
    def __init__(self, object_id, color, label_name, points, confidence=None, movable=True, segmentation_hex=None):
        self.object_id = object_id
        self.centroid = np.mean(points, axis=0)
        self.points = points
        self.label_name = label_name  # Now using label name instead of sem_label
        self.color = color
        self.segmentation_hex = segmentation_hex  # Original segmentation color from CSV
        self.movable = movable
        self.confidence = confidence if confidence is not None else 1.0
        self.visible = True
        self.update_hull_tree()
        self.compute_bounds()

        # lamp gets state attribute
        if label_name.lower() in ['lamp', 'light', 'lighting']:
            self.state = "state unknown"
    
    def compute_bounds(self):
        """Compute bounding box and geometric properties."""
        self.min_bound = np.min(self.points, axis=0)
        self.max_bound = np.max(self.points, axis=0)
        self.size = self.max_bound - self.min_bound
        self.volume = np.prod(self.size)
        
    def update_hull_tree(self):
        if len(self.points) >= 3:
            try:
                self.hull_tree = KDTree(self.points[ConvexHull(self.points).vertices])
            except:
                # Fallback if convex hull fails (e.g., points are coplanar)
                self.hull_tree = KDTree(self.points)
        else:
            self.hull_tree = KDTree(self.points)
    
    def is_point_inside(self, point, tolerance=0.05):
        """Check if a point is inside this object with some tolerance."""
        return (self.min_bound - tolerance <= point).all() and (point <= self.max_bound + tolerance).all()
    
    def get_overlap_ratio(self, other_node):
        """Calculate the overlap ratio with another node."""
        overlap_min = np.maximum(self.min_bound, other_node.min_bound)
        overlap_max = np.minimum(self.max_bound, other_node.max_bound)
        
        if (overlap_min < overlap_max).all():
            overlap_volume = np.prod(overlap_max - overlap_min)
            return overlap_volume / min(self.volume, other_node.volume)
        return 0.0
    
    def transform(self, transformation):
        """ Transform the points of the node using a translation, rotation, or homogeneous transformation matrix."""
        if isinstance(transformation, np.ndarray):
            if transformation.shape == (3,):
                self.centroid += transformation
                self.points += transformation
                self.update_hull_tree()
                self.compute_bounds()
            elif transformation.shape == (3, 3):
                self.points = np.dot(transformation, self.points.T).T
                self.centroid = np.dot(transformation, self.centroid)
                self.update_hull_tree()
                self.compute_bounds()
            elif transformation.shape == (4, 4):
                self.points = np.dot(transformation, np.vstack((self.points.T, np.ones(self.points.shape[0])))).T[:, :3]
                self.centroid = np.dot(transformation, np.append(self.centroid, 1))[:3]
                self.update_hull_tree()
                self.compute_bounds()
            else:
                raise ValueError("Invalid argument shape. Expected (3,) for translation, (3,3) for rotation, or (4,4) for homogeneous transformation.")
        else:
            raise TypeError("Invalid argument type. Expected numpy.ndarray.")

    def set_state(self, state):
        if state not in ["on", "off", "state unknown"]:
            raise ValueError("Invalid state. Expected 'on', 'off', or 'state unknown'.")
        self.state = state

    def get_segmentation_color(self):
        """Get the original segmentation color as RGB array."""
        if self.segmentation_hex:
            return hex_to_rgb(self.segmentation_hex)
        return self.color  # Fallback to detected color


class DrawerNode(ObjectNode):
    def __init__(self, object_id, color, label_name, points, confidence=1.0, movable=True, segmentation_hex=None):
        super().__init__(object_id, color, label_name, points, confidence, movable, segmentation_hex)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.equation, _ = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        self.box = None
        self.contains = []
    
    def sign_check(self, point):
        return np.dot(self.equation[:3], point) + self.equation[3] > 0
    
    def add_box(self, shelf_centroid):
        intersection = self.compute_intersection(shelf_centroid)
        
        bbox_points = []
        for point in self.points:
            bbox_points.append(point)
            bbox_points.append(point + 2* (shelf_centroid - intersection))

        points = np.array(bbox_points)

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(points)
          
        self.box = tmp_pcd.get_minimal_oriented_bounding_box()
    
    def compute_intersection(self, ray_start):
        signed_distance = (np.dot(self.equation[:3], ray_start) + self.equation[3]) / np.linalg.norm(self.equation[:3])
        
        if signed_distance > 0:
            direction = -self.equation[:3]  # Move in the negative normal direction
        else:
            direction = self.equation[:3]  # Move in the positive normal direction

        numerator = - (np.dot(self.equation[:3], ray_start) + self.equation[3])
        denominator = np.dot(self.equation[:3], direction)

        if denominator == 0:
            raise ValueError("The ray is parallel to the plane and does not intersect it.")
        
        t = numerator / denominator
        intersection_point = ray_start + t * direction

        return intersection_point
    
    def transform(self, transformation):
        super().transform(transformation)
        if self.box is not None and isinstance(transformation, np.ndarray):
            if transformation.shape == (3,):
                self.box.translate(transformation)
            elif transformation.shape == (4, 4):
                translation = transformation[:3, 3]
                rotation = transformation[:3, :3]
                self.box = self.box.rotate(rotation, center=np.array([0, 0, 0]))
                self.box.translate(translation)
        for node in self.contains:
            node.transform(transformation)


class LightSwitchNode(ObjectNode):
    def __init__(self, object_id, color, label_name, points, confidence=1.0, movable=True, segmentation_hex=None):
        super().__init__(object_id, color, label_name, points, confidence, movable, segmentation_hex)
        self.lamps = []
        self.button_count = "unknown"
        self.interaction = "unknown"
        self.normal = None

    def add_lamp(self, lamp_id):
        self.lamps.append(lamp_id)

    def set_button_count(self, count):
        self.button_count = count

    def set_interaction_type(self, interaction: str):
        self.interaction = interaction

    def set_normal(self, normal):
        self.normal = normal


class SimpleSceneGraph:
    def __init__(self, labels_csv_path, min_points=100, unmovable=[], pose=None, color_tolerance=0.01):
        self.index = 0
        self.nodes = dict()
        self.labels = dict()
        self.relationships = []  # List of SpatialRelationship objects
        self.tree = None
        self.ids = []
        self.min_points = min_points
        self.unmovable = unmovable
        self.pose = pose
        self.color_tolerance = color_tolerance
        
        # Load label mapping from CSV
        self.label_mapping = self.load_label_mapping(labels_csv_path)
        print(f"Loaded {len(self.label_mapping)} label mappings from {labels_csv_path}")

    def load_label_mapping(self, csv_path):
        """Load label mapping from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            if 'hex_color' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must contain 'hex_color' and 'label' columns")
            
            # Create mapping from hex color to label name
            mapping = {}
            for _, row in df.iterrows():
                hex_color = row['hex_color']
                if not hex_color.startswith('#'):
                    hex_color = f"#{hex_color}"
                
                # Store label with additional metadata if available
                label_info = {
                    'label': row['label'],
                    'confidence': row.get('similarity_score', 1.0),
                    'num_points': row.get('num_points', 0),
                    'mask_index': row.get('mask_index', -1),
                    'hex_color': hex_color  # Store the original hex color
                }
                mapping[hex_color.lower()] = label_info
            
            return mapping
        except Exception as e:
            print(f"Error loading label mapping from {csv_path}: {e}")
            return {}

    def add_node(self, color, label_name, points, confidence=1.0, segmentation_hex=None):
        """Add a node to the scene graph based on label name."""
        movable = label_name not in self.unmovable
        
        if not movable:
            self.nodes[self.index] = ObjectNode(self.index, color, label_name, points, confidence, movable=False, segmentation_hex=segmentation_hex)
        elif label_name.lower() in ['drawer', 'drawers']:
            self.nodes[self.index] = DrawerNode(self.index, color, label_name, points, confidence, segmentation_hex=segmentation_hex)
        elif label_name.lower() in ['light switch', 'lightswitch', 'switch']:
            self.nodes[self.index] = LightSwitchNode(self.index, color, label_name, points, confidence, segmentation_hex=segmentation_hex)
        else:
            self.nodes[self.index] = ObjectNode(self.index, color, label_name, points, confidence, segmentation_hex=segmentation_hex)
        
        self.labels.setdefault(label_name, []).append(self.index)
        self.ids.append(self.index)
        self.index += 1

    def detect_spatial_relationship(self, node1, node2, height_threshold=0.1, proximity_threshold=0.3, containment_threshold=0.3):
        """Detect the spatial relationship between two nodes."""
        
        # Calculate relative positions
        height_diff = node1.centroid[2] - node2.centroid[2]  # Assuming Z is up
        horizontal_dist = np.linalg.norm(node1.centroid[:2] - node2.centroid[:2])
        total_dist = np.linalg.norm(node1.centroid - node2.centroid)
        
        # Check for containment relationships
        overlap_ratio = node1.get_overlap_ratio(node2)
        
        # Containment: if one object is significantly inside another
        if overlap_ratio > containment_threshold:
            if node1.volume < node2.volume:
                return "in", 0.8
            else:
                return "contains", 0.8
        
        # On/Under relationships: significant height difference but close horizontally
        if abs(height_diff) > height_threshold and horizontal_dist < proximity_threshold:
            if height_diff > 0:
                return "on", 0.9  # node1 is on node2
            else:
                return "under", 0.9  # node1 is under node2
        
        # Next to: similar height, close proximity
        if abs(height_diff) < height_threshold and total_dist < proximity_threshold:
            return "next_to", 0.7
        
        return None, 0.0

    def detect_shared_surface_neighbors(self):
        """Detect objects that are next to each other because they share the same supporting surface."""
        
        # Group objects by their supporting surfaces
        surface_groups = {}
        
        for rel in self.relationships:
            if rel.relationship_type == "on":
                supporting_surface = rel.to_id
                if supporting_surface not in surface_groups:
                    surface_groups[supporting_surface] = []
                surface_groups[supporting_surface].append(rel.from_id)
        
        # For each surface, connect objects that are on it as "next_to"
        for surface_id, objects_on_surface in surface_groups.items():
            if len(objects_on_surface) > 1:
                for i, obj1_id in enumerate(objects_on_surface):
                    for obj2_id in objects_on_surface[i+1:]:
                        # Check if they're close enough to be considered "next to"
                        node1 = self.nodes[obj1_id]
                        node2 = self.nodes[obj2_id]
                        dist = np.linalg.norm(node1.centroid - node2.centroid)
                        
                        if dist < 0.6:  # Reduced threshold for shared surface neighbors
                            # Check if this relationship doesn't already exist
                            existing = any(r.from_id == obj1_id and r.to_id == obj2_id and r.relationship_type == "next_to" 
                                         for r in self.relationships)
                            if not existing:
                                rel1 = SpatialRelationship(obj1_id, obj2_id, "next_to", 0.6)
                                rel2 = SpatialRelationship(obj2_id, obj1_id, "next_to", 0.6)
                                self.relationships.extend([rel1, rel2])

    def detect_floor_standing_neighbors(self, proximity_threshold=1.0):
        """Detect 'next_to' relationships between floor-standing objects (furniture, etc.)."""
        
        # Find objects that are NOT "on" anything (i.e., standing on the floor)
        objects_on_other_things = set()
        for rel in self.relationships:
            if rel.relationship_type == "on":
                objects_on_other_things.add(rel.from_id)
        
        # Get floor-standing objects
        floor_standing_objects = []
        for obj_id, node in self.nodes.items():
            if obj_id not in objects_on_other_things:
                # Additional check: these should be larger objects (furniture-like)
                # Check if it's a furniture-type object based on label or size
                is_furniture = (
                    any(furniture_word in node.label_name.lower() 
                        for furniture_word in ['chair', 'table', 'cabinet', 'sofa', 'couch', 'desk', 
                                              'shelf', 'bookshelf', 'dresser', 'wardrobe', 'bed',
                                              'bench', 'stool', 'ottoman', 'nightstand', 'sideboard']) 
                    or node.volume > 0.1  # Or large enough volume (adjustable threshold)
                )
                
                if is_furniture:
                    floor_standing_objects.append(obj_id)
        
        print(f"Found {len(floor_standing_objects)} floor-standing objects: {[self.nodes[obj_id].label_name for obj_id in floor_standing_objects]}")
        
        # Connect nearby floor-standing objects as "next_to"
        for i, obj1_id in enumerate(floor_standing_objects):
            for obj2_id in floor_standing_objects[i+1:]:
                node1 = self.nodes[obj1_id]
                node2 = self.nodes[obj2_id]
                dist = np.linalg.norm(node1.centroid - node2.centroid)
                
                # Use a larger threshold for floor-standing furniture
                if dist < proximity_threshold:
                    # Check if this relationship doesn't already exist
                    existing = any(r.from_id == obj1_id and r.to_id == obj2_id and r.relationship_type == "next_to" 
                                 for r in self.relationships)
                    if not existing:
                        rel1 = SpatialRelationship(obj1_id, obj2_id, "next_to", 0.7)
                        rel2 = SpatialRelationship(obj2_id, obj1_id, "next_to", 0.7)
                        self.relationships.extend([rel1, rel2])
                        print(f"Added floor-standing 'next_to': {node1.label_name} <-> {node2.label_name} (dist: {dist:.2f})")

    def update_all_relationships(self):
        """Detect all spatial relationships between nodes."""
        self.relationships = []  # Reset relationships
        
        # Check all pairs of nodes
        for i, node1 in enumerate(self.nodes.values()):
            for j, node2 in enumerate(self.nodes.values()):
                if i >= j:  # Avoid duplicate pairs and self-comparison
                    continue
                
                rel_type, confidence = self.detect_spatial_relationship(node1, node2)
                
                if rel_type and confidence > 0.5:  # Only add confident relationships
                    # Add the relationship
                    rel = SpatialRelationship(node1.object_id, node2.object_id, rel_type, confidence)
                    self.relationships.append(rel)
                    
                    # Add the inverse relationship
                    inverse_rel = rel.get_inverse()
                    self.relationships.append(inverse_rel)
        
        # Detect shared surface neighbors
        self.detect_shared_surface_neighbors()
        
        # Detect floor-standing object neighbors
        self.detect_floor_standing_neighbors()
        
        print(f"Detected {len(self.relationships)} spatial relationships total")

    def get_relationships_for_node(self, node_id, relationship_type=None):
        """Get all relationships for a specific node."""
        relationships = [r for r in self.relationships if r.from_id == node_id]
        if relationship_type:
            relationships = [r for r in relationships if r.relationship_type == relationship_type]
        return relationships

    def get_relationship_summary(self):
        """Get a summary of all relationships."""
        summary = {}
        for rel in self.relationships:
            rel_type = rel.relationship_type
            if rel_type not in summary:
                summary[rel_type] = 0
            summary[rel_type] += 1
        return summary

    def init_graph(self):
        """Initialize graph with spatial relationships."""
        self.update_all_relationships()
        
        # Special processing for drawers
        for node in self.nodes.values():
            if isinstance(node, DrawerNode):
                # Find if this drawer has an "in" relationship with a shelf
                shelf_relationships = self.get_relationships_for_node(node.object_id, "in")
                if shelf_relationships:
                    shelf_id = shelf_relationships[0].to_id
                    node.add_box(self.nodes[shelf_id].centroid)

    def build_from_pointcloud(self, pointcloud_path):
        """Build scene graph from a color-labeled point cloud file."""
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        
        if len(pcd.points) == 0:
            raise ValueError(f"No points found in {pointcloud_path}")
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        if len(colors) == 0:
            print("Warning: No color information found in PLY file")
            # Create a single unknown object
            self.add_node(np.array([0.5, 0.5, 0.5]), "unknown", points)
        else:
            # Group points by color clusters
            print("Finding unique colors in point cloud...")
            unique_colors, color_labels = find_unique_colors(colors, tolerance=self.color_tolerance)
            print(f"Found {len(unique_colors)} unique colors")
            
            # Group points by color clusters
            for i, center_color in enumerate(unique_colors):
                mask = color_labels == i
                object_points = points[mask]
                
                # Convert color to hex for label lookup
                hex_color = rgb_to_hex(center_color).lower()
                
                # Skip white background objects (#ffffff)
                if hex_color == '#ffffff':
                    print(f"Skipping white background object with color {hex_color} ({len(object_points)} points)")
                    continue
                
                # Skip objects with too few points
                if len(object_points) < self.min_points:
                    continue
                
                # Look up label name from mapping
                label_info = self.label_mapping.get(hex_color, {
                    'label': f"unknown_{hex_color}", 
                    'confidence': 1.0, 
                    'num_points': len(object_points), 
                    'mask_index': -1,
                    'hex_color': hex_color
                })
                
                print(f"Creating object with color {hex_color} -> label '{label_info['label']}' ({len(object_points)} points, confidence: {label_info['confidence']:.3f})")
                
                self.add_node(center_color, label_info['label'], object_points, confidence=label_info['confidence'], segmentation_hex=label_info['hex_color'])
        
        # Initialize graph with relationships
        self.init_graph()
        
        # Build spatial index
        if len(self.ids) > 0:
            self.tree = KDTree(np.array([self.nodes[idx].centroid for idx in self.ids]))
        
        print(f"Built scene graph with {len(self.nodes)} objects")

    def get_node_info(self):
        """Print information about all nodes and their relationships."""
        for node in self.nodes.values():
            print(f"Object ID: {node.object_id}")
            print(f"Centroid: {node.centroid}")
            print(f"Label: {node.label_name}")
            print(f"Number of points: {len(node.points)}")
            print(f"Movable: {node.movable}")
            if hasattr(node, 'state'):
                print(f"State: {node.state}")
            
            # Print relationships
            relationships = self.get_relationships_for_node(node.object_id)
            if relationships:
                print("Relationships:")
                for rel in relationships:
                    other_node = self.nodes[rel.to_id]
                    print(f"  {rel.relationship_type} {other_node.label_name} (ID: {rel.to_id}, confidence: {rel.confidence:.2f})")
            print("---")

    def query(self, point):
        """Find the closest object to a given point."""
        if self.tree is None:
            return None
        _, idx = self.tree.query(point)
        return self.ids[idx]

    def get_distance(self, point):
        """Get distance to the closest object."""
        if self.tree is None:
            return float('inf')
        _, idx = self.tree.query(point)
        return np.linalg.norm(point - self.nodes[self.ids[idx]].centroid)

    def save(self, file_path):
        """Save the scene graph to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        """Load a scene graph from a pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def color_with_ibm_palette(self):
        """Apply IBM color palette to movable objects."""
        colors = np.array([
            [0.39215686, 0.56078431, 1.], 
            [0.47058824, 0.36862745, 0.94117647], 
            [0.8627451 , 0.14901961, 0.49803922],
            [0.99607843, 0.38039216, 0], 
            [1., 0.69019608, 0.], 
            [0.29803922, 0.68627451, 0.31372549], 
            [0., 0.6, 0.8],
            [0.70196078, 0.53333333, 1.], 
            [0.89803922, 0.22352941, 0.20784314], 
            [1., 0.25098039, 0.50588235]
        ])

        for node in self.nodes.values():
            if node.movable:
                node.color = colors[random.randint(0, len(colors)-1)]

    def visualize(self, centroids=True, connections=True, labels=False):
        """Visualize the scene graph with spatial relationships."""
        if len(self.nodes) == 0:
            print("No objects to visualize")
            return
        
        geometries = []
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 8  # Make edges thicker

        # Add object point clouds
        for node in self.nodes.values():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(node.points)
            # Use original segmentation color instead of detected color
            pcd_color = np.array(node.get_segmentation_color(), dtype=np.float64)
            pcd.paint_uniform_color(pcd_color)
            geometries.append((pcd, f"node_{node.object_id}", material))
            
            # Add big sphere at centroid - make it dark gray
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(node.centroid)
            sphere.paint_uniform_color([0.3, 0.3, 0.3])  # Dark gray
            geometries.append((sphere, f"sphere_{node.object_id}", material))

            # Add bounding boxes for drawers
            if isinstance(node, DrawerNode) and node.box is not None:
                geometries.append((node.box, f"bbox_{node.object_id}", line_mat))

        # Add centroids
        if centroids:
            centroid_pcd = o3d.geometry.PointCloud()
            centroids_xyz = np.array([node.centroid for node in self.nodes.values()])
            # Use original segmentation colors for centroids too
            centroids_colors = np.array([node.get_segmentation_color() for node in self.nodes.values()], dtype=np.float64)
            centroid_pcd.points = o3d.utility.Vector3dVector(centroids_xyz)
            centroid_pcd.colors = o3d.utility.Vector3dVector(centroids_colors)
            geometries.append((centroid_pcd, "centroids", material))

        # Add spatial relationship connections with light gray color and text labels
        if connections and len(self.relationships) > 0:
            # Use light gray for all relationships instead of different colors
            light_gray_color = [0.7, 0.7, 0.7]
            
            # Group relationships by type
            relationship_groups = {}
            for rel in self.relationships:
                if rel.relationship_type not in relationship_groups:
                    relationship_groups[rel.relationship_type] = []
                relationship_groups[rel.relationship_type].append(rel)
            
            # Create line sets for each relationship type (all in light gray)
            for rel_type, rels in relationship_groups.items():
                line_points = []
                line_indices = []
                idx = 0
                
                for rel in rels:
                    from_node = self.nodes[rel.from_id]
                    to_node = self.nodes[rel.to_id]
                    line_points.append(from_node.centroid)
                    line_points.append(to_node.centroid)
                    line_indices.append([idx, idx + 1])
                    idx += 2
                
                if line_points:
                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(line_points),
                        lines=o3d.utility.Vector2iVector(line_indices)
                    )
                    line_set.paint_uniform_color(light_gray_color)
                    geometries.append((line_set, f"relationships_{rel_type}", line_mat))

        # Create visualization window
        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Semantic Scene Graph - Press <S> to capture screenshot or <ESC> to quit", 1024, 1024)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        scene.scene.set_background(np.array([255.0, 255.0, 255.0, 1.0], dtype=np.float32))
        window.add_child(scene)

        # Add all geometries
        for geometry, name, mat in geometries:
            scene.scene.add_geometry(name, geometry, mat)

        # Set up camera
        if geometries:
            bounds = geometries[0][0].get_axis_aligned_bounding_box()
            for geometry, _, _ in geometries[1:]:
                bounds += geometry.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

        # Add labels if requested
        if labels:
            for node in self.nodes.values():
                label_text = node.label_name
                if not node.movable:
                    label_text = ""
                if hasattr(node, "state"):
                    label_text += f":\n {node.state}"
                
                # Add relationship info to labels
                relationships = self.get_relationships_for_node(node.object_id)
                if relationships:
                    rel_text = "\nRelations:\n" + "\n".join([f"{r.relationship_type} {self.nodes[r.to_id].label_name}" for r in relationships[:3]])
                    label_text += rel_text
                
                offset = np.array([0, 0, 0.05])
                scene.add_3d_label(node.centroid + offset, label_text)

        # Print relationship summary
        rel_summary = self.get_relationship_summary()
        print("\nRelationship Summary:")
        for rel_type, count in rel_summary.items():
            print(f"  {rel_type}: {count}")

        # Set up key event handlers
        def on_key_event(event):
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == gui.KeyName.S:  # Screenshot
                    image = gui.Application.instance.render_to_image(scene.scene, 1024, 1024)
                    current_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
                    filename = f"screenshot_{current_time}.png"
                    o3d.io.write_image(filename, image)
                    print(f"Screenshot saved as {filename}")
                    time.sleep(0.5)
                    return True
                if event.key == gui.KeyName.ESCAPE:  # Quit
                    gui.Application.instance.quit()
                    return True
            return False

        window.set_on_key(on_key_event)
        gui.Application.instance.run()


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build and visualize a semantic scene graph from a color-labeled point cloud.')
    parser.add_argument('--pointcloud', type=str, required=True, help='Path to the color-labeled point cloud file (mesh_labeled.ply)')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to the CSV file mapping hex colors to labels')
    parser.add_argument('--min_points', type=int, default=100, help='Minimum number of points required for an object')
    parser.add_argument('--color_tolerance', type=float, default=0.01, help='Tolerance for grouping similar colors')
    parser.add_argument('--unmovable', nargs='*', default=[], help='List of label names that should be unmovable')
    parser.add_argument('--save', type=str, help='Path to save the scene graph pickle file')
    parser.add_argument('--visualize', action='store_true', help='Visualize the scene graph')
    parser.add_argument('--labels', action='store_true', help='Show labels in visualization')
    
    args = parser.parse_args()
    
    # Create scene graph
    sg = SimpleSceneGraph(
        labels_csv_path=args.labels_csv,
        min_points=args.min_points,
        unmovable=args.unmovable,
        color_tolerance=args.color_tolerance
    )
    
    # Build from point cloud
    sg.build_from_pointcloud(args.pointcloud)
    
    # Apply colors
    sg.color_with_ibm_palette()
    
    # Print info
    sg.get_node_info()
    
    # Save if requested
    if args.save:
        sg.save(args.save)
        print(f"Scene graph saved to {args.save}")
    
    # Visualize if requested
    if args.visualize:
        sg.visualize(labels=args.labels) 