import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud')
        
        self.declare_parameter('depth_topic', '/realsense/depth/image')
        self.declare_parameter('info_topic', '/realsense/depth/camera_info')
        self.declare_parameter('cloud_topic', '/depth_cloud')
        
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('info_topic').value
        cloud_topic = self.get_parameter('cloud_topic').value
        
        self.bridge = CvBridge()
        self.camera_info = None
        
        self.sub_info = self.create_subscription(
            CameraInfo, 
            info_topic, 
            self.info_callback, 
            10
        )
        
        self.sub_depth = self.create_subscription(
            Image, 
            depth_topic, 
            self.depth_callback, 
            10
        )
        
        self.pub_cloud = self.create_publisher(PointCloud2, cloud_topic, 10)
        self.get_logger().info(f"Depth to PointCloud node started. Listening to {depth_topic}")

    def info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        if self.camera_info is None:
            return

        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # Get intrinsic parameters
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Handle encoding
        if msg.encoding == '16UC1':
            depth = cv_image.astype(np.float32) / 1000.0
        elif msg.encoding == '32FC1':
            depth = cv_image
        else:
            self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
            return

        height, width = depth.shape
        
        # Create meshgrid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth
        valid = (depth > 0.1) & (depth < 10.0) # 10m max range
        
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        
        # Stack points
        points = np.stack((x, y, z), axis=-1)
        
        # Create PointCloud2
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id
        
        # Use simple creation
        # points is (N, 3)
        # We need structured array or flat list
        # pc2.create_cloud_xyz32 takes header and points as iterable of lists/tuples
        
        # Using create_cloud_xyz32 is efficient enough for Python if points is not huge
        # But for 640x480 it might be slow.
        # Let's try it. If slow, we optimize.
        
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_cloud.publish(cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
