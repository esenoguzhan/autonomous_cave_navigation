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
            cv_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding) # Changed desired_encoding
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # Handle encoding
        if msg.encoding == '16UC1':
            depth = cv_image.astype(np.float32) / 1000.0
        elif msg.encoding == '32FC1':
            depth = cv_image
        else:
            self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
            return

        height, width = depth.shape
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Diagnostic logging (first frame only)
        if getattr(self, '_first_frame', True):
            self.get_logger().info(f"Depth Camera Info: res={width}x{height}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            self._first_frame = False

        # Create meshgrid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth and downsample (stride of 2)
        skip = 2
        depth_sub = depth[::skip, ::skip]
        valid = (depth_sub > 0.1) & (depth_sub < 15.0)
        
        if not np.any(valid):
            return

        z = depth_sub[valid]
        x = (u[::skip, ::skip][valid] - cx) * z / fx
        y = (v[::skip, ::skip][valid] - cy) * z / fy
        
        # Stack points
        points = np.stack((x, y, z), axis=-1)
        
        # Create PointCloud2 message
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id
        pc_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_cloud.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
