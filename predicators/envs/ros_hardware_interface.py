import logging
import time
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64 # Assuming this might be needed elsewhere, keeping import
import shlex
from subprocess import Popen
import numpy as np
import warnings
from typing import Optional, Tuple

# Constants (can be overridden by parameters)
DEFAULT_MAX_CARTESIAN_VELOCITY = 0.2 # m/s
DEFAULT_MAX_ROTATION_VELOCITY = 0.5 # rad/s
DEFAULT_GRIPPER_ACTION_DURATION = 1.5 # seconds
DEFAULT_NODE_NAME = "ros_hardware_interface"
PANDA_FINGER_JOINT1_NAME = "panda_finger_joint1"
PANDA_FINGER_JOINT2_NAME = "panda_finger_joint2"

class ROSHardwareInterface:
    """Handles ROS communication for interacting with robot hardware."""

    def __init__(self,
                node_name: str = DEFAULT_NODE_NAME,
                robot_base_frame: Optional[str] = None,
                ee_pose_topic: Optional[str] = None,
                twist_command_topic: Optional[str] = None,
                gripper_state_topic: Optional[str] = None,
                gripper_open_cmd: str = 'rosrun franka_interactive_controllers franka_gripper_run_node 1',
                gripper_close_cmd: str = 'rosrun franka_interactive_controllers franka_gripper_run_node 0',
                wait_for_connections_duration: float = 5.0,
                gripper_action_duration: float = DEFAULT_GRIPPER_ACTION_DURATION,
                max_cartesian_velocity: float = DEFAULT_MAX_CARTESIAN_VELOCITY,
                max_rotation_velocity: float = DEFAULT_MAX_ROTATION_VELOCITY,
                init_gripper_open: bool = True):
        """
        Initializes the ROS hardware interface.

        Args:
            node_name: Name for the ROS node.
            robot_base_frame: TF frame ID of the robot's base. Reads from ROS param ~robot_base_frame if None.
            ee_pose_topic: Topic for the end-effector pose (PoseStamped). Reads from ROS param ~ee_pose_topic if None.
            twist_command_topic: Topic to publish Twist commands. Reads from ROS param ~twist_command_topic if None.
            gripper_state_topic: Topic for gripper joint states (JointState). Reads from ROS param ~gripper_state_topic if None.
            gripper_open_cmd: Shell command to open the gripper.
            gripper_close_cmd: Shell command to close the gripper.
            wait_for_connections_duration: Time (s) to wait for ROS connections/TF buffer.
            gripper_action_duration: Expected duration (s) for gripper open/close action.
            max_cartesian_velocity: Max linear velocity (m/s) for scaling actions.
            max_rotation_velocity: Max angular velocity (rad/s) for scaling actions.
            init_gripper_open: If True, attempts to open the gripper upon initialization.
        """
        # Initialize ROS node (only if one doesn't exist)
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

            rospy.loginfo(f"ROS node '{node_name}' initialized by ROSHardwareInterface.")
        else:
            rospy.loginfo(f"ROS node '{rospy.core.get_node_uri()}' already initialized.")

        # --- Store configurations ---
        self._gripper_open_cmd = gripper_open_cmd
        self._gripper_close_cmd = gripper_close_cmd
        self._gripper_action_duration = rospy.Duration(gripper_action_duration)
        self.max_cartesian_velocity = max_cartesian_velocity
        self.max_rotation_velocity = max_rotation_velocity

        # --- ROS Parameters (Get defaults if not provided) ---
        # Frame IDs
        self.robot_base_frame = robot_base_frame or rospy.get_param("~robot_base_frame", "panda_link0")
        # Topic Names
        self.ee_pose_topic = ee_pose_topic or rospy.get_param("~ee_pose_topic", "/franka_state_controller/ee_uni_gripper_pose")
        self.twist_command_topic = twist_command_topic or rospy.get_param("~twist_command_topic", "/passiveDS/desired_twist")
        self.gripper_state_topic = gripper_state_topic or rospy.get_param("~gripper_state_topic", "/franka_gripper/joint_states")
        # Motion Capture Subscriber
        self.door_pose_topic = rospy.get_param("~door_pose_topic", "/natnet_ros/Door/pose") # Check type! Assuming PoseStamped
        # self.cabinet_pose_topic = rospy.get_param("~cabinet_pose_topic", "/natnet_ros/Cabinet/pose") # Check type! Assuming PoseStamped
        self.object_pose_topic = rospy.get_param("~object_pose_topic", "/natnet_ros/musturd/pose") # Check type! Assuming PoseStamped
        self.gripper_alt_pose_topic = rospy.get_param("~gripper_alt_pose_topic", "/natnet_ros/franka_gripper/pose") # Check type! Assuming PoseStamped
        # New mocap object topics
        self.bowl_pose_topic = rospy.get_param("~bowl_pose_topic", "/natnet_ros/bowl/pose")
        self.lid_pose_topic = rospy.get_param("~lid_pose_topic", "/natnet_ros/lid/pose")
        self.pan_pose_topic = rospy.get_param("~pan_pose_topic", "/natnet_ros/pan/pose")
        self.dishrack_pose_topic = rospy.get_param("~dishrack_pose_topic", "/natnet_ros/dishrack/pose")
        self.mug_pose_topic = rospy.get_param("~mug_pose_topic", "/natnet_ros/mug/pose")
        self.banana_pose_topic = rospy.get_param("~banana_pose_topic", "/natnet_ros/banana/pose")

        # --- ROS Communication ---
        self.tf_listener = tf.TransformListener()
        # Publishers
        self.twist_pub = rospy.Publisher(self.twist_command_topic, Twist, queue_size=1)
        # Subscribers
        self._current_ee_pose_msg: Optional[PoseStamped] = None
        self._current_gripper_joint_state: Optional[JointState] = None
        self._current_door_pose_msg: Optional[PoseStamped] = None
        self._current_cabinet_pose_msg: Optional[PoseStamped] = None
        self._current_object_pose_msg: Optional[PoseStamped] = None
        self._current_gripper_alt_pose_msg: Optional[PoseStamped] = None
        # New mocap object pose messages
        self._current_bowl_pose_msg: Optional[PoseStamped] = None
        self._current_lid_pose_msg: Optional[PoseStamped] = None
        self._current_pan_pose_msg: Optional[PoseStamped] = None
        self._current_dishrack_pose_msg: Optional[PoseStamped] = None
        self._current_mug_pose_msg: Optional[PoseStamped] = None
        self._current_banana_pose_msg: Optional[PoseStamped] = None

        # Subscribers
        self._door_pose_sub = rospy.Subscriber(self.door_pose_topic, PoseStamped, self._door_pose_callback, queue_size=1)
        # self._cabinet_pose_sub = rospy.Subscriber(self.cabinet_pose_topic, PoseStamped, self._cabinet_pose_callback, queue_size=1)
        self._object_pose_sub = rospy.Subscriber(self.object_pose_topic, PoseStamped, self._object_pose_callback, queue_size=1)
        self._gripper_alt_pose_sub = rospy.Subscriber(self.gripper_alt_pose_topic, PoseStamped, self._gripper_alt_pose_callback, queue_size=1)
        # New mocap object subscribers
        self._bowl_pose_sub = rospy.Subscriber(self.bowl_pose_topic, PoseStamped, self._bowl_pose_callback, queue_size=1)
        self._lid_pose_sub = rospy.Subscriber(self.lid_pose_topic, PoseStamped, self._lid_pose_callback, queue_size=1)
        self._pan_pose_sub = rospy.Subscriber(self.pan_pose_topic, PoseStamped, self._pan_pose_callback, queue_size=1)
        self._dishrack_pose_sub = rospy.Subscriber(self.dishrack_pose_topic, PoseStamped, self._dishrack_pose_callback, queue_size=1)
        self._mug_pose_sub = rospy.Subscriber(self.mug_pose_topic, PoseStamped, self._mug_pose_callback, queue_size=1)
        self._banana_pose_sub = rospy.Subscriber(self.banana_pose_topic, PoseStamped, self._banana_pose_callback, queue_size=1)

        self._ee_pose_sub = rospy.Subscriber(self.ee_pose_topic, PoseStamped, self._ee_pose_callback, queue_size=1)
        self._gripper_state_sub = rospy.Subscriber(self.gripper_state_topic, JointState, self._gripper_state_callback, queue_size=1)

        # --- Internal State ---
        self._desired_gripper_command: Optional[float] = None # 0.0 for close, 1.0 for open
        self._last_gripper_command_time: rospy.Time = rospy.Time(0) # Initialize to epoch

        # Allow time for connections and TF buffer
        rospy.loginfo(f"Waiting {wait_for_connections_duration}s for ROS connections and TF buffer...")
        rospy.sleep(wait_for_connections_duration)
        rospy.loginfo("ROS connections established.")

        # Initialize gripper if requested
        if init_gripper_open:
            rospy.loginfo("Sending initial open command to gripper...")
            self.set_gripper_state(1.0) # Command open
            rospy.loginfo(f"Gripper initialization: Waiting {self._gripper_action_duration.to_sec()}s for completion...")
            rospy.sleep(self._gripper_action_duration)
            rospy.loginfo("Gripper initialized.")

    # --- ROS Callbacks ---
    def _door_pose_callback(self, msg: PoseStamped):
        """Store the latest door pose message."""
        self._current_door_pose_msg = msg

    def _cabinet_pose_callback(self, msg: PoseStamped):
        """Store the latest cabinet pose message."""
        self._current_cabinet_pose_msg = msg

    def _object_pose_callback(self, msg: PoseStamped):
        """Store the latest object pose message."""
        self._current_object_pose_msg = msg

    def _gripper_alt_pose_callback(self, msg: PoseStamped):
        """Store the latest gripper alternate pose message."""
        self._current_gripper_alt_pose_msg = msg

    # New mocap object callbacks
    def _bowl_pose_callback(self, msg: PoseStamped):
        """Store the latest bowl pose message."""
        self._current_bowl_pose_msg = msg

    def _lid_pose_callback(self, msg: PoseStamped):
        """Store the latest lid pose message."""
        self._current_lid_pose_msg = msg

    def _pan_pose_callback(self, msg: PoseStamped):
        """Store the latest pan pose message."""
        self._current_pan_pose_msg = msg

    def _dishrack_pose_callback(self, msg: PoseStamped):
        """Store the latest dishrack pose message."""
        self._current_dishrack_pose_msg = msg

    def _mug_pose_callback(self, msg: PoseStamped):
        """Store the latest mug pose message."""
        self._current_mug_pose_msg = msg

    def _banana_pose_callback(self, msg: PoseStamped):
        """Store the latest banana pose message."""
        self._current_banana_pose_msg = msg

    def _ee_pose_callback(self, msg: PoseStamped):
        """Store the latest end-effector pose message."""
        self._current_ee_pose_msg = msg

    def _gripper_state_callback(self, msg: JointState):
        """Store the latest gripper state message."""
        self._current_gripper_joint_state = msg

    # --- Getters for Sensor Data ---
    def get_ee_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received end-effector PoseStamped message.

        Args:
            wait_for_message: If True, waits until the first message is received (up to timeout).
            timeout: Max time (s) to wait if wait_for_message is True.

        Returns:
            The latest PoseStamped message, or None if no message received (and not waiting/timeout).
        """
        if wait_for_message and self._current_ee_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.ee_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_ee_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_ee_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.ee_pose_topic}")
                return None
        return self._current_ee_pose_msg

    def get_gripper_state(self) -> Optional[JointState]:
        """Returns the latest received gripper JointState message."""
        # Could add wait logic similar to get_ee_pose if needed
        return self._current_gripper_joint_state

    def get_gripper_positions(self) -> Optional[Tuple[float, float]]:
        """Returns the positions of the gripper fingers, if available."""
        joint_state = self.get_gripper_state()
        if joint_state is not None and len(joint_state.position) >= 2:
            # Assuming first two positions are left/right fingers
            return (joint_state.position[0], joint_state.position[1])
        return None

    def transform_pose(self, pose_stamped: PoseStamped, target_frame: str, timeout: float = 0.5) -> Optional[PoseStamped]:
        """
        Transforms a PoseStamped message to the target frame using TF.

        Args:
            pose_stamped: The input PoseStamped message.
            target_frame: The desired target TF frame.
            timeout: Time (s) to wait for the TF transform.

        Returns:
            The transformed PoseStamped message, or None if transform fails.
        """
        try:
            source_frame = pose_stamped.header.frame_id
            time_stamp = pose_stamped.header.stamp
            self.tf_listener.waitForTransform(target_frame, source_frame, time_stamp, rospy.Duration(timeout))
            pose_transformed = self.tf_listener.transformPose(target_frame, pose_stamped)
            return pose_transformed
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr_throttle(2.0, f"TF Error transforming pose from '{source_frame}' to '{target_frame}': {e}")
            return None
        except Exception as e:
             rospy.logerr(f"Unexpected error during TF transform: {e}")
             return None

    def get_door_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received door PoseStamped message from mocap.

        Args:
            wait_for_message: If True, waits until the first message is received (up to timeout).
            timeout: Max time (s) to wait if wait_for_message is True.

        Returns:
            The latest PoseStamped message, or None if no message received (and not waiting/timeout).
        """
        if wait_for_message and self._current_door_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.door_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_door_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_door_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.door_pose_topic}")
                return None
        return self._current_door_pose_msg

    def get_cabinet_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received cabinet PoseStamped message from mocap.

        Args:
            wait_for_message: If True, waits until the first message is received (up to timeout).
            timeout: Max time (s) to wait if wait_for_message is True.

        Returns:
            The latest PoseStamped message, or None if no message received (and not waiting/timeout).
        """
        if wait_for_message and self._current_cabinet_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.cabinet_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_cabinet_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_cabinet_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.cabinet_pose_topic}")
                return None
        return self._current_cabinet_pose_msg

    def get_gripper_alt_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received gripper alternate PoseStamped message from mocap.

        Args:
            wait_for_message: If True, waits until the first message is received (up to timeout).
            timeout: Max time (s) to wait if wait_for_message is True.

        Returns:
            The latest PoseStamped message, or None if no message received (and not waiting/timeout).
        """
        if wait_for_message and self._current_gripper_alt_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.gripper_alt_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_gripper_alt_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_gripper_alt_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.gripper_alt_pose_topic}")
                return None
        return self._current_gripper_alt_pose_msg
    
    def get_object_pose(self) -> Optional[PoseStamped]:
        """
        Returns the latest received object PoseStamped message from mocap.
        """
        return self._current_object_pose_msg

    # New getter methods for mocap objects
    def get_bowl_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received bowl PoseStamped message from mocap.
        """
        if wait_for_message and self._current_bowl_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.bowl_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_bowl_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_bowl_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.bowl_pose_topic}")
                return None
        return self._current_bowl_pose_msg

    def get_lid_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received lid PoseStamped message from mocap.
        """
        if wait_for_message and self._current_lid_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.lid_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_lid_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_lid_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.lid_pose_topic}")
                return None
        return self._current_lid_pose_msg

    def get_pan_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received pan PoseStamped message from mocap.
        """
        if wait_for_message and self._current_pan_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.pan_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_pan_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_pan_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.pan_pose_topic}")
                return None
        return self._current_pan_pose_msg

    def get_dishrack_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received dishrack PoseStamped message from mocap.
        """
        if wait_for_message and self._current_dishrack_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.dishrack_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_dishrack_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_dishrack_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.dishrack_pose_topic}")
                return None
        return self._current_dishrack_pose_msg

    def get_mug_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received mug PoseStamped message from mocap.
        """
        if wait_for_message and self._current_mug_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.mug_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_mug_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_mug_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.mug_pose_topic}")
                return None
        return self._current_mug_pose_msg

    def get_banana_pose(self, wait_for_message: bool = True, timeout: float = 5.0) -> Optional[PoseStamped]:
        """
        Returns the latest received banana PoseStamped message from mocap.
        """
        if wait_for_message and self._current_banana_pose_msg is None:
            rospy.loginfo_throttle(1.0, f"Waiting for first message on {self.banana_pose_topic}...")
            wait_start_time = rospy.Time.now()
            wait_duration = rospy.Duration(timeout)
            while self._current_banana_pose_msg is None and (rospy.Time.now() - wait_start_time) < wait_duration and not rospy.is_shutdown():
                rospy.sleep(0.05)
            if self._current_banana_pose_msg is None:
                rospy.logerr(f"Timeout waiting for message on {self.banana_pose_topic}")
                return None
        return self._current_banana_pose_msg
            

    # --- Robot Commands ---
    def publish_twist_command(self, linear_vel: np.ndarray, angular_vel: np.ndarray):
        """
        Publishes a Twist command based on desired linear and angular velocities.

        Args:
            linear_vel: 3D numpy array for linear velocity (x, y, z).
            angular_vel: 3D numpy array for angular velocity (x, y, z).
        """
        if not isinstance(linear_vel, np.ndarray) or linear_vel.shape != (3,):
            warnings.warn(f"Linear velocity must be a 3D numpy array, got: {linear_vel}")
            return
        if not isinstance(angular_vel, np.ndarray) or angular_vel.shape != (3,):
            warnings.warn(f"Angular velocity must be a 3D numpy array, got: {angular_vel}")
            return

        twist_msg = Twist()
        twist_msg.linear.x = linear_vel[0]
        twist_msg.linear.y = linear_vel[1]
        twist_msg.linear.z = linear_vel[2]
        twist_msg.angular.x = angular_vel[0]
        twist_msg.angular.y = angular_vel[1]
        twist_msg.angular.z = angular_vel[2]
        rospy.loginfo_throttle(1.0, f"Publishing twist command: {linear_vel}, {angular_vel}")
        self.twist_pub.publish(twist_msg)

    def scale_and_publish_twist_action(self, action_velocities: np.ndarray):
        """
        Scales velocity components from an action array [-1, 1] and publishes Twist.

        Args:
            action_velocities: 6D numpy array [vx, vy, vz, wx, wy, wz] scaled between -1 and 1.
        """
        if not isinstance(action_velocities, np.ndarray) or action_velocities.shape != (6,):
            warnings.warn(f"Action velocities must be a 6D numpy array, got: {action_velocities}")
            return

        linear_vel = action_velocities[:3] * self.max_cartesian_velocity
        angular_vel = action_velocities[3:6] * self.max_rotation_velocity
        self.publish_twist_command(linear_vel, angular_vel)

    def _execute_gripper_command(self, command: str) -> bool:
        """Executes a shell command for the gripper and handles the process."""
        try:
            rospy.loginfo(f"Executing gripper command: {command}")
            node_process = Popen(shlex.split(command))
            # We don't wait here in the interface, the caller manages delays
            # based on _gripper_action_duration if needed.
            # Giving a very short time for the process to potentially fail immediately.
            time.sleep(0.1)
            if node_process.poll() is not None: # Check if process terminated early (error)
                 rospy.logerr(f"Gripper command process terminated unexpectedly: {command}")
                 return False
            # The caller should manage termination if needed, or let it run.
            # For the franka_gripper_run_node, it likely terminates itself.
            # If the command was long-running, Popen object could be returned.
            return True
        except Exception as e:
            rospy.logerr(f"Failed to execute gripper command '{command}': {e}")
            return False

    def set_gripper_state(self, desired_state: float):
        """
        Sends a command to open or close the gripper if the state changed
        and enough time has passed since the last command.

        Args:
            desired_state: 0.0 for close, 1.0 for open.
        """
        # Check if enough time has passed since the last command
        if rospy.Time.now() - self._last_gripper_command_time < self._gripper_action_duration:
            # rospy.logdebug("Gripper command interval not met, skipping.")
            return

        command_executed = False
        if desired_state < 0.5 and (self._desired_gripper_command is None or self._desired_gripper_command < 0.5): # Command Open
             rospy.loginfo("Commanding gripper OPEN")
             if self._execute_gripper_command(self._gripper_open_cmd):
                 self._desired_gripper_command = 1.0
                 command_executed = True
        elif desired_state >= 0.5 and (self._desired_gripper_command is None or self._desired_gripper_command >= 0.5): # Command Close
             rospy.loginfo("Commanding gripper CLOSE")
             if self._execute_gripper_command(self._gripper_close_cmd):
                 self._desired_gripper_command = 0.0
                 command_executed = True
        # else: # Debugging
             # rospy.logdebug(f"Gripper command {desired_state} matches desired internal state {self._desired_gripper_command}, not sending.")

        if command_executed:
            self._last_gripper_command_time = rospy.Time.now()

    def stop_motion(self):
        """Publishes a zero Twist command to stop robot motion."""
        rospy.loginfo("Publishing zero twist to stop motion.")
        self.publish_twist_command(np.zeros(3), np.zeros(3))

    def shutdown(self):
        """Clean up ROS connections."""
        rospy.loginfo("Shutting down ROS Hardware Interface.")
        self.stop_motion() # Send zero twist on shutdown
        if hasattr(self, '_ee_pose_sub'): self._ee_pose_sub.unregister()
        if hasattr(self, '_gripper_state_sub'): self._gripper_state_sub.unregister()
        if hasattr(self, 'twist_pub'): self.twist_pub.unregister()
        # Unregister mocap subscribers
        if hasattr(self, '_door_pose_sub'): self._door_pose_sub.unregister()
        if hasattr(self, '_object_pose_sub'): self._object_pose_sub.unregister()
        if hasattr(self, '_gripper_alt_pose_sub'): self._gripper_alt_pose_sub.unregister()
        # Unregister new mocap subscribers
        if hasattr(self, '_bowl_pose_sub'): self._bowl_pose_sub.unregister()
        if hasattr(self, '_lid_pose_sub'): self._lid_pose_sub.unregister()
        if hasattr(self, '_pan_pose_sub'): self._pan_pose_sub.unregister()
        if hasattr(self, '_dishrack_pose_sub'): self._dishrack_pose_sub.unregister()
        if hasattr(self, '_mug_pose_sub'): self._mug_pose_sub.unregister()
        if hasattr(self, '_banana_pose_sub'): self._banana_pose_sub.unregister()
        # Note: Does not shut down the rospy node itself, as other parts of
        # the application might still be using ROS.

# Example Usage (Optional - for testing this file directly)
if __name__ == '__main__':
    try:
        rospy.loginfo("Initializing ROSHardwareInterface for standalone test...")
        # Example: Override some parameters if needed
        interface = ROSHardwareInterface(
            node_name="hardware_interface_test",
            # ee_pose_topic="/alternate/ee_pose", # Example override
            init_gripper_open=True # Start with gripper open
        )
        rospy.loginfo("ROSHardwareInterface initialized.")

        rate = rospy.Rate(10) # 10 Hz
        start_time = rospy.Time.now()
        duration = rospy.Duration(10.0) # Run test for 10 seconds

        while not rospy.is_shutdown() and (rospy.Time.now() - start_time) < duration:
            # --- Get Data ---
            ee_pose = interface.get_ee_pose(wait_for_message=False) # Don't block loop if msg not ready
            gripper_joints = interface.get_gripper_state()
            gripper_pos = interface.get_gripper_positions()

            if ee_pose:
                 # Optional: Transform pose
                 # pose_in_base = interface.transform_pose(ee_pose, interface.robot_base_frame)
                 # if pose_in_base:
                 #    rospy.loginfo_throttle(1.0, f"EE Pose (Base): {pose_in_base.pose.position.x:.3f}, {pose_in_base.pose.position.y:.3f}, {pose_in_base.pose.position.z:.3f}")
                 # else:
                 rospy.loginfo_throttle(1.0, f"EE Pose ({ee_pose.header.frame_id}): {ee_pose.pose.position.x:.3f}, {ee_pose.pose.position.y:.3f}, {ee_pose.pose.position.z:.3f}")

            if gripper_pos:
                rospy.loginfo_throttle(1.0, f"Gripper Positions: L={gripper_pos[0]:.4f}, R={gripper_pos[1]:.4f}")
            else:
                rospy.loginfo_throttle(1.0, "Waiting for gripper state...")

            # --- Send Commands (Example: alternate open/close every 2 seconds) ---
            elapsed_secs = (rospy.Time.now() - start_time).to_sec()
            if int(elapsed_secs / 2) % 2 == 0:
                # Command open for the first 2s, 4-6s, 8-10s intervals
                interface.set_gripper_state(1.0) # Open command
            else:
                # Command close for 2-4s, 6-8s intervals
                interface.set_gripper_state(0.0) # Close command

            # Example: Send a small circular motion command
            # t = elapsed_secs
            # linear = np.array([0.0, 0.05 * np.sin(t), 0.0])
            # angular = np.array([0.0, 0.0, 0.1 * np.cos(t)])
            # interface.publish_twist_command(linear, angular)

            # Example: Scale action from [-1, 1] range
            # action = np.array([0.0, 0.5 * np.sin(t), 0.0, 0.0, 0.0, 0.2 * np.cos(t)]) # Example 6D action
            # interface.scale_and_publish_twist_action(action)


            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        if 'interface' in locals():
            interface.shutdown()
        rospy.loginfo("Hardware interface test finished.") 