import numpy as np
import copy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial import KDTree
from generate_scene_graph import SimpleSceneGraph, SpatialRelationship
import argparse


class SceneGraphUpdater:
    def __init__(self, scene_graph):
        """Initialize with an existing scene graph."""
        self.scene_graph = scene_graph
        self.original_scene_graph = None  # Will store the state before updates
    
    def find_objects_by_label(self, label_name):
        """Find all objects with a given label name (case insensitive partial match)."""
        matching_objects = []
        label_lower = label_name.lower()
        
        for obj_id, node in self.scene_graph.nodes.items():
            if label_lower in node.label_name.lower():
                matching_objects.append(obj_id)
        
        return matching_objects
    
    def translate_object(self, direction, distance_cm, object_label):
        """
        Translate an object in the specified direction.
        
        Args:
            direction: "left" or "right" 
            distance_cm: distance in centimeters
            object_label: label name of the object to move
        """
        # Find objects with matching label
        matching_objects = self.find_objects_by_label(object_label)
        
        if not matching_objects:
            print(f"No objects found with label containing '{object_label}'")
            return False
        
        if len(matching_objects) > 1:
            print(f"Multiple objects found with label '{object_label}': {[self.scene_graph.nodes[obj_id].label_name for obj_id in matching_objects]}")
            print(f"Moving all {len(matching_objects)} objects")
        
        # Convert distance to meters (assuming input is in cm)
        distance_m = distance_cm / 100.0
        
        # Determine translation vector
        if direction.lower() == "left":
            translation = np.array([0, -distance_m, 0])  # Negative Y
        elif direction.lower() == "right":
            translation = np.array([0, distance_m, 0])   # Positive Y
        else:
            print(f"Invalid direction '{direction}'. Use 'left' or 'right'")
            return False
        
        # Apply translation to all matching objects
        for obj_id in matching_objects:
            node = self.scene_graph.nodes[obj_id]
            print(f"Translating '{node.label_name}' (ID: {obj_id}) {direction} by {distance_cm}cm")
            
            # Transform the node (this updates points, centroid, bounds, etc.)
            node.transform(translation)
        
        # Update spatial relationships after movement
        print("Updating spatial relationships...")
        self.scene_graph.update_all_relationships()
        
        # Rebuild spatial index
        if len(self.scene_graph.ids) > 0:
            self.scene_graph.tree = KDTree(np.array([self.scene_graph.nodes[idx].centroid for idx in self.scene_graph.ids]))
        
        return True
    
    def remove_object(self, object_label):
        """
        Remove an object completely from the scene graph.
        
        Args:
            object_label: label name of the object to remove
        """
        # Find objects with matching label
        matching_objects = self.find_objects_by_label(object_label)
        
        if not matching_objects:
            print(f"No objects found with label containing '{object_label}'")
            return False
        
        if len(matching_objects) > 1:
            print(f"Multiple objects found with label '{object_label}': {[self.scene_graph.nodes[obj_id].label_name for obj_id in matching_objects]}")
            print(f"Removing all {len(matching_objects)} objects")
        
        # Remove each matching object
        for obj_id in matching_objects:
            node = self.scene_graph.nodes[obj_id]
            print(f"Removing '{node.label_name}' (ID: {obj_id})")
            
            # Remove from nodes dictionary
            del self.scene_graph.nodes[obj_id]
            
            # Remove from labels dictionary
            if node.label_name in self.scene_graph.labels:
                if obj_id in self.scene_graph.labels[node.label_name]:
                    self.scene_graph.labels[node.label_name].remove(obj_id)
                # Remove the label entry if no objects left with this label
                if not self.scene_graph.labels[node.label_name]:
                    del self.scene_graph.labels[node.label_name]
            
            # Remove from ids list
            if obj_id in self.scene_graph.ids:
                self.scene_graph.ids.remove(obj_id)
        
        # Remove all relationships involving these objects
        original_relationships = self.scene_graph.relationships[:]
        self.scene_graph.relationships = []
        
        for rel in original_relationships:
            if rel.from_id not in matching_objects and rel.to_id not in matching_objects:
                self.scene_graph.relationships.append(rel)
        
        # Rebuild spatial index
        if len(self.scene_graph.ids) > 0:
            from scipy.spatial import KDTree
            self.scene_graph.tree = KDTree(np.array([self.scene_graph.nodes[idx].centroid for idx in self.scene_graph.ids]))
        else:
            self.scene_graph.tree = None
        
        print(f"Removed {len(matching_objects)} object(s). Remaining objects: {len(self.scene_graph.nodes)}")
        return True
    
    def apply_action(self, action):
        """
        Apply a single action to the scene graph.
        
        Args:
            action: List representing an action, e.g., ["translate", "left", "20", "White plastic bottle"]
        """
        if not isinstance(action, list) or len(action) < 2:
            print(f"Invalid action format: {action}")
            return False
        
        action_type = action[0].lower()
        
        if action_type == "translate":
            if len(action) != 4:
                print(f"Invalid translate action. Expected 4 arguments, got {len(action)}: {action}")
                return False
            
            direction = action[1]
            try:
                distance = float(action[2])
            except ValueError:
                print(f"Invalid distance value: {action[2]}")
                return False
            
            object_label = action[3]
            return self.translate_object(direction, distance, object_label)
        
        elif action_type == "remove":
            if len(action) != 2:
                print(f"Invalid remove action. Expected 2 arguments, got {len(action)}: {action}")
                return False
            
            object_label = action[1]
            return self.remove_object(object_label)
        
        else:
            print(f"Unknown action type: {action_type}")
            return False
    
    def apply_actions(self, actions):
        """
        Apply a list of actions to the scene graph.
        
        Args:
            actions: List of action lists
        """
        print(f"Applying {len(actions)} actions...")
        print("=" * 50)
        
        for i, action in enumerate(actions):
            print(f"\nAction {i+1}: {action}")
            success = self.apply_action(action)
            if not success:
                print(f"Failed to apply action: {action}")
            print("-" * 30)
    
    def save_state_as_original(self):
        """Save the current state as the original state for before/after comparison."""
        self.original_scene_graph = copy.deepcopy(self.scene_graph)
        print("Saved current state as original for before/after comparison")
    
    def visualize_before_after(self, title_suffix=""):
        """Visualize both original and updated scene graphs side by side."""
        if self.original_scene_graph is None:
            print("No original state saved. Showing current state only.")
            self.scene_graph.visualize()
            return
        
        # Create two separate visualizations
        print("\n" + "="*50)
        print("BEFORE (Original State)")
        print("="*50)
        self.original_scene_graph.get_node_info()
        
        print("\n" + "="*50)
        print("AFTER (Updated State)")
        print("="*50)
        self.scene_graph.get_node_info()
        
        # Show original first
        print("\nShowing BEFORE state - Close window to continue to AFTER state...")
        self.original_scene_graph.visualize()
        
        # Show updated
        print("Showing AFTER state...")
        self.scene_graph.visualize()


def main():
    parser = argparse.ArgumentParser(description='Update and visualize changes to a semantic scene graph.')
    parser.add_argument('--pointcloud', type=str, required=True, help='Path to the color-labeled point cloud file')
    parser.add_argument('--labels_csv', type=str, required=True, help='Path to the CSV file mapping hex colors to labels')
    parser.add_argument('--min_points', type=int, default=100, help='Minimum number of points required for an object')
    parser.add_argument('--color_tolerance', type=float, default=0.01, help='Tolerance for grouping similar colors')
    parser.add_argument('--unmovable', nargs='*', default=[], help='List of label names that should be unmovable')
    parser.add_argument('--actions', nargs='*', help='Actions to apply, each action should be space-separated arguments')
    parser.add_argument('--save_before', type=str, help='Path to save the original scene graph')
    parser.add_argument('--save_after', type=str, help='Path to save the updated scene graph')
    
    args = parser.parse_args()
    
    # Create and build initial scene graph
    print("Building initial scene graph...")
    sg = SimpleSceneGraph(
        labels_csv_path=args.labels_csv,
        min_points=args.min_points,
        unmovable=args.unmovable,
        color_tolerance=args.color_tolerance
    )
    
    sg.build_from_pointcloud(args.pointcloud)
    sg.color_with_ibm_palette()
    
    print(f"\nInitial scene graph built with {len(sg.nodes)} objects")
    
    # Create updater
    updater = SceneGraphUpdater(sg)
    
    # Save original state
    updater.save_state_as_original()
    
    # Save original if requested
    if args.save_before:
        sg.save(args.save_before)
        print(f"Original scene graph saved to {args.save_before}")
    
    # Parse and apply actions
    if args.actions:
        # Parse actions from command line arguments
        # Example: --actions translate left 20 "White plastic bottle" remove "White plastic bottle"
        actions = []
        i = 0
        while i < len(args.actions):
            if args.actions[i] == "translate" and i + 3 < len(args.actions):
                action = [args.actions[i], args.actions[i+1], args.actions[i+2], args.actions[i+3]]
                actions.append(action)
                i += 4
            elif args.actions[i] == "remove" and i + 1 < len(args.actions):
                action = [args.actions[i], args.actions[i+1]]
                actions.append(action)
                i += 2
            else:
                print(f"Invalid action starting at: {args.actions[i]}")
                i += 1
        
        if actions:
            updater.apply_actions(actions)
    else:
        # Example actions for demonstration
        example_actions = [
            ["translate", "left", "20", "bottle"],
            ["remove", "bottle"]
        ]
        
        print("No actions provided. Running example actions:")
        for action in example_actions:
            print(f"  {action}")
        
        response = input("\nApply these example actions? (y/n): ")
        if response.lower() == 'y':
            updater.apply_actions(example_actions)
    
    # Save updated scene graph if requested
    if args.save_after:
        sg.save(args.save_after)
        print(f"Updated scene graph saved to {args.save_after}")
    
    # Show before/after visualization
    print("\nShowing before/after visualization...")
    updater.visualize_before_after()


if __name__ == "__main__":
    main()
