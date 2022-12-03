from gym import Env
from gym.spaces import Discrete, Box
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque

import tensorflow as tf
import sys
import math
import random
import time
import os.path
import numpy as np
sys.path.append('MacAPI')
import sim

np.set_printoptions(threshold=sys.maxsize)

# Max movement along X
low, high = -0.05, 0.05

MODEL_NAME = 'model1'

# Create folder to store model
if not os.path.isdir('models'):
    os.makedirs('models')

GAMMA = 0.99 # Discount rate
LEARNING_RATE = 0.001
REPLAY_MEMORY_SIZE = 50000 # Remember 50 episodes
MIN_REPLAY_MEMORY_SIZE = 1000 # Minimum number of steps in memory to start training
MINIBATCH_SIZE = 64 # How many samples from memory to use for training
UPDATE_TARGET_EVERY = 5 # How many episodes to update target

EPISODES = 300

# Exploration values
epsilon = 1 # Not constant, will be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Stats
AGGREGATE_STATS_EVERY = 50 # Episodes
ep_rewards = [-200]

def get_distance_3d(a, b):
    a_x, a_y, a_z = a[0], a[1], a[2]
    b_x, b_y, b_z = b[0], b[1], b[2]

    # Distance between source cup and receive cup
    return math.sqrt((a_x - b_x)**2 + (a_y - b_y)**2 + (a_z - b_z)**2)

class ModifiedTensorBoard(TensorBoard):        
    # Override init for initial step and writer (to have one log file for all .fit() call)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    # Override, saves logs with step number
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Override, train one batch only, no need to save at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Override, so writer doesn't close
    def on_train_end(self, _):
        pass

    # Creates method for saving metrics
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()

class DQNAgent:
    def __init__(self, state_size, action_size):
        '''Gym environment variables'''
        self.state_size = state_size # 7 -> source_x, and x,y,z for each cube
        self.action_size = action_size # 5 -> [-2, -1, 0, 1, 2]

        # [-2, -1, 0, 1, 2]
        self.action_space = Discrete(action_size)
        # Observation space bounds
        self.source_x_low = -0.95 # -0.85 + low
        self.source_x_high = -0.75 # -0.85 + high
        self.cube1_low = [-1.85, -1, 0]
        self.cube1_high = [0.15, 1, 1]
        self.cube2_low = [-1.85, -1, 0]
        self.cube2_high = [0.15, 1, 1]

        self.lower_bounds = np.array([self.source_x_low, self.cube1_low[0], self.cube1_low[1], self.cube1_low[2],
                                self.cube2_low[0], self.cube2_low[1], self.cube2_low[2]])
        self.upper_bounds = np.array([self.source_x_high, self.cube1_high[0], self.cube1_high[1], self.cube1_high[2],
                                self.cube2_high[0], self.cube2_high[1], self.cube2_high[2]])

        # Initialize observation space
        self.observation_space = Box(self.lower_bounds, self.upper_bounds, dtype=np.float32) 

        '''Simulator variables'''
        self.state = None
        self.total_frames = 1000
        #j in range(velReal.shape[0])
        self.current_frame = 0
        self.speed = None

        self.clientID = None
        self.source_cup_handle = None
        self.receive_cup_handle = None
        self.cubes_handles = []
        
        self.source_cup_position = None
        self.receive_cup_position = None
        self.cubes_positions = []
        self.center_position = None
        self.joint_position = None

        '''DQN variables/hyperparameters'''
        # Main model
        self.model = self._build_model()
        # Target model
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        # Remember last n steps of training as an array
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        #self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Counter for when to update target network with main network weights
        self.target_update_counter = 0

    def train(self):
        global epsilon

        for e in range(1, EPISODES+1):
            print(f"Episode {e}:")

            # Update tensorboard step every episode
            #self.tensorboard.step = e

            # Reset episode reward and step number every episode
            episode_reward = 0
            step = 1

            # Set rotation velocity randomly
            rng = np.random.default_rng()
            velReal = self.rotation_velocity(rng)

            # Start simulation, process objects/handles
            self.start_simulation()

            # Set initial position of the source cup and initialize state
            state = self.reset(rng)
            # Reshape state for Keras
            #state = np.reshape(state, [1, self.state_size])

            done = False

            for j in range(velReal.shape[0]):
                self.step_chores()
                # Initialize the speed of the source cup at this frame
                self.speed = velReal[j]

                # Pick next action, greedy epsilon
                action = self.act(state)

                # Take next action
                new_state, reward, done, _ = self.step(action)

                # Count reward
                episode_reward += reward

                # Save current time step into deque
                self.remember((state, action, reward, new_state, done))
                # Train main network
                self.train_batch(done, step)

                # Update state variable
                state = new_state
                step += 1
                
                # Break if cup goes back to vertical position
                if done:
                    break
            #end for (current episode)

            # Stop simulation
            self.stop_simulation()

            # Append episode reward to rewards list and log stats 
            # after AGGREGATE_STATS_EVERY number of episodes
            ep_rewards.append(episode_reward)
            if not e % AGGREGATE_STATS_EVERY or e == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                #self.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
                print(f'average_reward from episode {e-5} to {e}: {average_reward}\nepsilon: {epsilon}')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        #end for (total episodes)

        # Insane save at the very end, hope it saves
        self.save(f"models/{MODEL_NAME}.model")


    def run(self):
        # Load model from training
        self.model = load_model(f"models/{MODEL_NAME}.model")

        for e in range(1):
            print(f"Episode {e+1}:")

            # Set rotation velocity randomly
            rng = np.random.default_rng()
            velReal = self.rotation_velocity(rng)

            # Start simulation, process objects/handles
            self.start_simulation()

            # Set initial position of the source cup and initialize state
            state = self.reset(rng)
            state = np.reshape(state, [1, self.state_size])

            done = False

            for j in range(velReal.shape[0]):
                self.step_chores()
                # Initialize the speed of the source cup at this frame
                self.speed = velReal[j]

                # Pick next action based on weights
                action = self.eval_act(state)

                # Take next action
                next_state, reward, done, _ = self.step(action)
            
                #print(f"State: {state}, Reward: {reward}, Action: {action}")

                next_state = np.reshape(next_state, [1, self.state_size]) # Reshape state for Keras

                # Update state variable
                state = next_state
                
                # Break if cup goes back to vertical position
                if done:
                    break
            #end for

    def step(self, action):
        '''
        Move the simulation forward a frame and calculate the reward for the action taken
        '''
        dist = self.get_reward()
        # Rotate cup based on speed value
        self.rotate_cup()
        # Move cup laterally based on selected action (minus 2 for indexing)
        self.move_cup(action-2)

        # Calculate reward as the negative distance between the source up and the receiving cup
        new_dist = self.get_reward()

        # If new distance is less than old -> positive reward, else negative reward
        reward = dist - new_dist
        #if new_dist == dist and action != 2: # Action selected was anything but 0 and source cup did not move
        #    reward = -0.05
        #    print(f"reward for not moving due to limited range: {reward}, (action selected: {action-2})")
        #print(f"reward: {reward}")

        # Get the position of both cubes
        for cube, i in zip(self.cubes_handles, range(0, 2)):
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions[i] = cube_position

        # Check if episode is finished
        done = bool(self.current_frame > 10 and self.joint_position > 0)

        # Keep track of current frame number
        self.current_frame += 1

        info = {}
        # Get state after step completed, to return
        self.state = np.array([self.source_cup_position[0], self.cubes_positions[0][0], self.cubes_positions[0][1], 
                                self.cubes_positions[0][2], self.cubes_positions[1][0], self.cubes_positions[1][1],
                                self.cubes_positions[1][2]])

        #Return step info
        return self.state, reward, done, info
    
    def reset(self, rng):
        '''
        Reset the simulation and all relevant variables before the beginning of each episode.
        '''
        # Reset source cup position 
        self.cubes_handles = []
        self.cubes_positions = []
        # Sets source cup randomly, initializes self.cubes_positions
        self.set_random_cup_position(rng)
        # Current frame is j
        self.current_frame = 0
        # Speed is velReal[j]
        self.speed = 0
        self.joint_position = None

        # Update state for new episode
        self.state = np.array([self.source_cup_position[0], self.cubes_positions[0][0], self.cubes_positions[0][1], 
                                self.cubes_positions[0][2], self.cubes_positions[1][0], self.cubes_positions[1][1],
                                self.cubes_positions[1][2]])

        return self.state

    def get_reward(self):
        reward_ = -get_distance_3d(self.receive_cup_position, self.source_cup_position)
        max_d = -get_distance_3d(self.receive_cup_position, [-0.8500, -0.1555, 0.7248])
        if self.center_position > -0.8500:
            min_d = -get_distance_3d(self.receive_cup_position, 
                                    [self.center_position+high, -0.1555, 0.7248])
        else:
            min_d = -get_distance_3d(self.receive_cup_position, 
                                    [self.center_position+low, -0.1555, 0.7248])
        '''
        flag = 0
        for cube in self.cubes_positions:
            # If cubes are within x bounds of receiving cup rim
            # and above the table
            if (-0.88 < cube[0] < -0.82) and (0.30 < cube[2] < 0.80):
                flag += 1
        '''
                
        # Both cubes are lined up along x-axis with receiving cup
        '''if flag == 2:
            reward = 1000 - self.normalize(reward_, min_d, max_d, 100)
            #print(f"reward HIT: {reward}")
        else: 
            reward = 0 - self.normalize(reward_, min_d, max_d, 100)
            #print(f"reward MISS: {reward}")
        '''

        #if max_d < reward_ < min_d:
        #    print(f"{min_d}, {max_d}, {reward_}")
        #print(f"reward: {self.normalize(reward_, min_d, max_d, 1)}")

        reward = self.normalize(reward_, min_d, max_d, 1)

        return reward

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    # Store step information to memory
    # (state, action, reward, new_state, done)
    def remember(self, transition):
        self.memory.append(transition)
    
    def save(self, name):
        self.model.save(name)

    def train_batch(self, terminal_state, step):
        if len(self.memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Get a minibatch of random samples from memory
        minibatch = random.sample(self.memory, MINIBATCH_SIZE)

        # Get current states from batch, query model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from batch, query model for Q values
        # When target network is being used, query it, otherwise main network queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Enumerate batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not terminal state, get new Q from future states, otherwise set to 0
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward
            
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Append to training data
            X.append(current_state)
            y.append(current_qs)
        
        #Fit on all samples as a batch, log terminal state only
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter per episode
        if terminal_state:
            self.target_update_counter += 1
        
        # If counter reaches UPDATE_TARGET_EVERY, update target with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
    
    def act(self, state):
        if np.random.random() > epsilon:
            return np.argmax(self.get_qs(state))
        else:
            return np.random.randint(0, self.action_size)
    
    def eval_act(self, state):
        act_values = self.model.predict(state)
        print(f"Selecting action based on values: {act_values[0]}\nAction selected: {np.argmax(act_values[0]) - 2}")
        return np.argmax(act_values[0])
        
    def set_random_cup_position(self, rng):
        '''
        Set the source cup position randomly before each episode.
        Also grabs the positions of the items in the scene.
        '''
        # Get source cup position before random move
        returnCode, self.source_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_buffer)
        #print(f'Source Cup Initial Position:{self.source_cup_position}')

        # Move cup along x axis
        global low, high
        move_x = low + (high - low) * rng.random()
        self.source_cup_position[0] = self.source_cup_position[0] + move_x

        returnCode = sim.simxSetObjectPosition(self.clientID, self.source_cup_handle, -1, self.source_cup_position,
                                            sim.simx_opmode_blocking)
        self.triggerSim()

        returnCode, self.source_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_buffer)
        #print(f'Pouring cup randomly set position:{self.source_cup_position}')

        returnCode, self.receive_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.receive_cup_handle, -1, sim.simx_opmode_buffer)
        #print(f'Receiving cup position:{self.receive_cup_position}')

        obj_type = "Cuboid"
        number_of_blocks = 2
        for cube in range(number_of_blocks):
            res, obj_handle = sim.simxGetObjectHandle(self.clientID,
                                                    f'{obj_type}{cube}',
                                                    sim.simx_opmode_blocking)
            self.cubes_handles.append(obj_handle)
        
        # Get the starting position of cubes
        for cube in self.cubes_handles:
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions.append(cube_position)

        self.wait_()

        # Get center position of source cup for bounds calculations
        self.center_position = self.source_cup_position[0]

    def start_simulation(self):
        ''' Function to communicate with Coppelia Remote API and start the simulation '''
        sim.simxFinish(-1)  # Just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1', 19000, True, True, 5000,
                                5)  # Connect to CoppeliaSim
        if self.clientID != -1:
            print('Connected to remote API server')
        else:
            print("fail")
            sys.exit()

        returnCode = sim.simxSynchronous(self.clientID, True)
        returnCode = sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

        if returnCode != 0 and returnCode != 1:
            print("something is wrong")
            print(returnCode)
            exit(0)

        self.triggerSim()

        # Get the handle for the source container
        res, self.source_cup_handle = sim.simxGetObjectHandle(self.clientID, 'joint',
                                            sim.simx_opmode_blocking)
        res, self.receive_cup_handle = sim.simxGetObjectHandle(self.clientID, 'receive',
                                            sim.simx_opmode_blocking)
        # Start streaming the data
        returnCode, original_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_streaming)
        returnCode, original_position = sim.simxGetObjectPosition(
            self.clientID, self.receive_cup_handle, -1, sim.simx_opmode_streaming)
        returnCode, original_position = sim.simxGetJointPosition(
            self.clientID, self.source_cup_handle, sim.simx_opmode_streaming)

        # Get object handles
        self.get_cubes()

    def stop_simulation(self):
        ''' Function to stop the episode '''
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)
        print("Simulation stopped.")

    def get_cubes(self):
        # Drop blocks in source container
        self.triggerSim()
        number_of_blocks = 2
        #print('Initial number of blocks=', number_of_blocks)
        self.setNumberOfBlocks(blocks=number_of_blocks,
                                typeOf='cube',
                                mass=0.002,
                                blockLength=0.025,
                                frictionCube=0.06,
                                frictionCup=0.8)

        self.triggerSim()

        # Get handles of cubes created
        obj_type = "Cuboid"
        for cube in range(number_of_blocks):
            res, obj_handle = sim.simxGetObjectHandle(self.clientID,
                                                    f'{obj_type}{cube}',
                                                    sim.simx_opmode_blocking)
            self.cubes_handles.append(obj_handle)

        self.triggerSim()

        # Get the starting position of cubes
        for cube in self.cubes_handles:
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions.append(cube_position)
        
        # Give time for the cubes to finish falling
        self.wait_()

    def rotation_velocity(self, rng):
        ''' 
        Set rotation velocity randomly, rotation velocity is a composition of two sinusoidal velocities 
        '''
        # Sinusoidal velocity
        forward = [-0.3, -0.35, -0.4, -0.45, -0.50, -0.55, -0.60, -0.65]
        backward = [0.75, 0.8, 0.85, 0.90]
        freq = 60
        ts = np.linspace(0, 1000 / freq, 1000)
        velFor = rng.choice(forward) * np.sin(2 * np.pi * 1 / 20 * ts)
        velBack = rng.choice(backward) * np.sin(2 * np.pi * 1 / 10 * ts)
        velSin = velFor
        idxFor = np.argmax(velFor > 0)
        velSin[idxFor:] = velBack[idxFor:]
        velReal = velSin
        return velReal

    def step_chores(self):
        '''
        Simulation processing before each step.
        '''
        # 60Hz
        self.triggerSim()
        # Make sure simulation step finishes
        returnCode, pingTime = sim.simxGetPingTime(self.clientID)

    def triggerSim(self):
        e = sim.simxSynchronousTrigger(self.clientID)
        step_status = 'successful' if e == 0 else 'error'

    def setNumberOfBlocks(self, blocks, typeOf, mass, blockLength,
                      frictionCube, frictionCup):
        '''
            Function to set the number of blocks in the simulation.
        '''
        emptyBuff = bytearray()
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
            self.clientID, 'Table', sim.sim_scripttype_childscript, 'setNumberOfBlocks',
            [blocks], [mass, blockLength, frictionCube, frictionCup], [typeOf],
            emptyBuff, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print('Results: ', retStrings)  # display the reply from CoppeliaSim (in this case, the handle of the created dummy)
        else:
            print('Remote function call failed')

    def rotate_cup(self):
        ''' 
        Function to rotate cup
        '''
        errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.source_cup_handle, self.speed,
                                                sim.simx_opmode_oneshot)
        returnCode, self.joint_position = sim.simxGetJointPosition(self.clientID, self.source_cup_handle,
                                                        sim.simx_opmode_buffer)

    def move_cup(self, action):
        ''' 
        Function to move the pouring cup laterally during the rotation. 
        '''
        global low, high
        resolution = 0.001
        move_x = resolution * action
        movement = self.source_cup_position[0] + move_x
        if self.center_position + low < movement < self.center_position + high:
            self.source_cup_position[0] = movement
            returnCode = sim.simxSetObjectPosition(self.clientID, self.source_cup_handle, -1,
                                                self.source_cup_position,
                                                sim.simx_opmode_blocking)

    def wait_(self):
        for _ in range(60):
            self.triggerSim()

    def normalize(self, val, min_val, max_val, q):
        '''
        norm_i = (x_i - min(x)) / max(x) - min(x)) * Q
        Normalize values between 0 and Q
        '''
        norm_val = (val - min_val) / (max_val - min_val) * q
        return norm_val

class CubesCups(Env):
    def __init__(self, num_episodes=300,
                min_lr=0.1, min_epsilon=0.1, 
                discount=0.99, decay=25):

        #100 for source_x, 100 for x,y,z of each cube
        self.bins = (100, 100, 100, 100, 100, 100, 100)
        # [-2, -1, 0, 1, 2]
        self.action_space = Discrete(5)
        # Observation space bounds
        self.source_x_low = -0.95 # -0.85 + low
        self.source_x_high = -0.75 # -0.85 + high
        self.cube1_x_low, self.cube1_y_low, self.cube1_z_low = -1.85, -1, 0
        self.cube1_x_high, self.cube1_y_high, self.cube1_z_high = 0.15, 1, 1
        self.cube2_x_low, self.cube2_y_low, self.cube2_z_low = -1.85, -1, 0
        self.cube2_x_high, self.cube2_y_high, self.cube2_z_high = 0.15, 1, 1
        #self.velReal_low = -0.7361215932167728
        #self.velReal_high = 0.8499989492543077 

        self.lower_bounds = np.array([self.source_x_low, self.cube1_x_low, self.cube1_y_low, self.cube1_z_low,
                                self.cube2_x_low, self.cube2_y_low, self.cube2_z_low])
        self.upper_bounds = np.array([self.source_x_high, self.cube1_x_high, self.cube1_y_high, self.cube1_z_high,
                                self.cube2_x_high, self.cube2_y_high, self.cube2_z_high])

        # Initialize observation space
        self.observation_space = Box(self.lower_bounds, self.upper_bounds, dtype=np.float32) 

        # Initialize Q-table
        #self.q_table = np.zeros(self.bins + (self.action_space.n,))

        self.state = None
        self.total_frames = 1000
        #j in range(velReal.shape[0])
        self.current_frame = 0
        self.speed = None

        self.clientID = None
        self.source_cup_handle = None
        self.receive_cup_handle = None
        self.cubes_handles = []
        
        self.source_cup_position = None
        self.receive_cup_position = None
        self.cubes_positions = []
        self.center_position = None
        self.joint_position = None

        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

    def step(self, action):
        '''
        Move the simulation forward a frame and calculate the reward for the action taken
        '''
        # Calculate reward as the negative distance between the source up and the receiving cup
        reward = -get_distance_3d(self.source_cup_position, self.receive_cup_position)
    
        # Rotate cup based on speed value
        self.rotate_cup()
        # Move cup laterally based on selected action in Q-table
        self.move_cup(action)

        # Get the position of both cubes
        for cube, i in zip(self.cubes_handles, range(0, 2)):
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions[i] = cube_position

        # Check if episode is finished
        done = bool(self.current_frame > 10 and self.joint_position > 0)

        # Keep track of current frame number
        self.current_frame += 1

        info = {}
        # Get state after step completed, to return
        self.state = np.array([self.source_cup_position[0], self.speed])

        #Return step info
        return self.state, reward, done, info
    
    def reset(self, rng):
        '''
        Reset the simulation and all relevant variables before the beginning of each episode.
        '''
        # Reset source cup position 
        self.cubes_handles = []
        self.cubes_positions = []
        self.set_random_cup_position(rng)
        # Current frame is j
        self.current_frame = 0
        # Speed is velReal[j]
        self.speed = 0
        self.joint_position = None

        # Update state for new episode
        self.state = np.array([self.source_cup_position[0], self.speed])

        return self.state

    def train(self):
        rewards_all_episodes = []
        rewards_filename = "rewards_history.txt"
        q_table_filename = "q_table.npy"

        for e in range(self.num_episodes):
            print(f"Episode {e+1}:")

            # Set rotation velocity randomly
            rng = np.random.default_rng()
            velReal = self.rotation_velocity(rng)

            # Start simulation, process objects/handles
            self.start_simulation()

            # Set initial position of the source cup and initialize state
            state = self.discretize_state(self.reset(rng))

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False
            # Keep track of rewards for each episode, to be stored in a list
            rewards_current_episode = 0

            for j in range(velReal.shape[0]):
                #
                self.step_chores()
                # Initialize the speed of the source cup at this frame
                self.speed = velReal[j]

                # Pick next action, greedy epsilon
                action = self.pick_action(state)

                # Take next action
                obs, reward, done, _ = self.step(action)

                # Normalize speed value for q_table
                self.speed = self.normalize(velReal[j], self.velReal_low, self.velReal_high)
                # Calculate new state, continuous -> discrete
                new_state = self.discretize_state(obs)
            
                # Update Q-table for Q(s,a)
                print(f"State: {state}, Reward: {reward}, Action: {action}")
                self.update_q(state, action, new_state, reward)

                # Update state variable
                state = new_state

                # Keep track of rewards for current episode
                rewards_current_episode += reward
                
                # Break if cup goes back to vertical position
                if done:
                    break
            #end for

            # Stop simulation
            self.stop_simulation()

            # Append current episode's reward to total rewards list for later
            rewards_all_episodes.append(rewards_current_episode)

        # Save the Q-table to a .npy file
        with open(q_table_filename, 'wb') as f:
            np.save(f, self.q_table)
        print(f"Q-table saved to {q_table_filename}")

        # Append this episodes rewards to a .txt file
        with open(rewards_filename, 'wb') as f:
            np.savetxt(f, rewards_all_episodes)
        print(f"Saved rewards history to {rewards_filename}")

    def run(self):
        t = 0
        done = False
        q_table_filename = "q_table.npy"

        # Load q_table.pkl for updating, if it exists
        if(os.path.exists(q_table_filename)):
            with open(q_table_filename, 'rb') as f:
                self.q_table = np.load(f)
            print("Q-table loaded.")

            print(self.q_table)
        
        # Set rotation velocity randomly
        rng = np.random.default_rng()
        velReal = self.rotation_velocity(rng)

        # Start simulation, process objects/handles
        self.start_simulation()

        # Set initial position of the source cup and initialize state
        state = self.discretize_state(self.reset(rng))

        for j in range(velReal.shape[0]):
            self.step_chores()
            # Initialize the speed of the source cup at this frame
            self.speed = velReal[j]

            # Pick next action, greedy epsilon
            action = np.argmax(self.q_table[state]) - 2

            # Take next action
            obs, reward, done, _ = self.step(action)

            # Calculate new state, continuous -> discrete
            new_state = self.discretize_state(obs)

            print(f"State: {state}, Action: {action}")
        
            # Update state variable
            state = new_state
            
            # Break if cup goes back to vertical position
            if done:
                break
        #end for
        
        # Stop simulation
        self.stop_simulation()
        
        return t

    def start_simulation(self):
        ''' Function to communicate with Coppelia Remote API and start the simulation '''
        sim.simxFinish(-1)  # Just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1', 19000, True, True, 5000,
                                5)  # Connect to CoppeliaSim
        if self.clientID != -1:
            print('Connected to remote API server')
        else:
            print("fail")
            sys.exit()

        returnCode = sim.simxSynchronous(self.clientID, True)
        returnCode = sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

        if returnCode != 0 and returnCode != 1:
            print("something is wrong")
            print(returnCode)
            exit(0)

        self.triggerSim()

        # Get the handle for the source container
        res, self.source_cup_handle = sim.simxGetObjectHandle(self.clientID, 'joint',
                                            sim.simx_opmode_blocking)
        res, self.receive_cup_handle = sim.simxGetObjectHandle(self.clientID, 'receive',
                                            sim.simx_opmode_blocking)
        # Start streaming the data
        returnCode, original_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_streaming)
        returnCode, original_position = sim.simxGetObjectPosition(
            self.clientID, self.receive_cup_handle, -1, sim.simx_opmode_streaming)
        returnCode, original_position = sim.simxGetJointPosition(
            self.clientID, self.source_cup_handle, sim.simx_opmode_streaming)

        # Get object handles
        self.get_cubes()

    def stop_simulation(self):
        ''' Function to stop the episode '''
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)
        print("Simulation stopped.")

    def get_cubes(self):
        # Drop blocks in source container
        self.triggerSim()
        number_of_blocks = 2
        print('Initial number of blocks=', number_of_blocks)
        self.setNumberOfBlocks(blocks=number_of_blocks,
                                typeOf='cube',
                                mass=0.002,
                                blockLength=0.025,
                                frictionCube=0.06,
                                frictionCup=0.8)

        self.triggerSim()

        # Get handles of cubes created
        obj_type = "Cuboid"
        for cube in range(number_of_blocks):
            res, obj_handle = sim.simxGetObjectHandle(self.clientID,
                                                    f'{obj_type}{cube}',
                                                    sim.simx_opmode_blocking)
            self.cubes_handles.append(obj_handle)

        self.triggerSim()

        # Get the starting position of cubes
        for cube in self.cubes_handles:
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions.append(cube_position)
        
        # Give time for the cubes to finish falling
        self.wait_()

    def rotation_velocity(self, rng):
        ''' 
        Set rotation velocity randomly, rotation velocity is a composition of two sinusoidal velocities 
        '''
        # Sinusoidal velocity
        forward = [-0.3, -0.35, -0.4, -0.45, -0.50, -0.55, -0.60, -0.65]
        backward = [0.75, 0.8, 0.85, 0.90]
        freq = 60
        ts = np.linspace(0, 1000 / freq, 1000)
        velFor = rng.choice(forward) * np.sin(2 * np.pi * 1 / 20 * ts)
        velBack = rng.choice(backward) * np.sin(2 * np.pi * 1 / 10 * ts)
        velSin = velFor
        idxFor = np.argmax(velFor > 0)
        velSin[idxFor:] = velBack[idxFor:]
        velReal = velSin
        return velReal
    
    def set_random_cup_position(self, rng):
        '''
        Set the source cup position randomly before each episode.
        Also grabs the positions of the items in the scene.
        '''
        # Get source cup position before random move
        returnCode, self.source_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_buffer)
        print(f'Source Cup Initial Position:{self.source_cup_position}')

        # Move cup along x axis
        global low, high
        move_x = low + (high - low) * rng.random()
        self.source_cup_position[0] = self.source_cup_position[0] + move_x

        returnCode = sim.simxSetObjectPosition(self.clientID, self.source_cup_handle, -1, self.source_cup_position,
                                            sim.simx_opmode_blocking)
        self.triggerSim()

        returnCode, self.source_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.source_cup_handle, -1, sim.simx_opmode_buffer)
        print(f'Pouring cup randomly set position:{self.source_cup_position}')

        returnCode, self.receive_cup_position = sim.simxGetObjectPosition(
            self.clientID, self.receive_cup_handle, -1, sim.simx_opmode_buffer)
        print(f'Receiving cup position:{self.receive_cup_position}')

        obj_type = "Cuboid"
        number_of_blocks = 2
        for cube in range(number_of_blocks):
            res, obj_handle = sim.simxGetObjectHandle(self.clientID,
                                                    f'{obj_type}{cube}',
                                                    sim.simx_opmode_blocking)
            self.cubes_handles.append(obj_handle)
        
        # Get the starting position of cubes
        for cube in self.cubes_handles:
            returnCode, cube_position = sim.simxGetObjectPosition(
                self.clientID, cube, -1, sim.simx_opmode_streaming)
            self.cubes_positions.append(cube_position)

        self.wait_()

        # Get center position of source cup for bounds calculations
        self.center_position = self.source_cup_position[0]

    def step_chores(self):
        '''
        Simulation processing before each step.
        '''
        # 60Hz
        self.triggerSim()
        # Make sure simulation step finishes
        returnCode, pingTime = sim.simxGetPingTime(self.clientID)

    def triggerSim(self):
        e = sim.simxSynchronousTrigger(self.clientID)
        step_status = 'successful' if e == 0 else 'error'

    def setNumberOfBlocks(self, blocks, typeOf, mass, blockLength,
                      frictionCube, frictionCup):
        '''
            Function to set the number of blocks in the simulation.
        '''
        emptyBuff = bytearray()
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
            self.clientID, 'Table', sim.sim_scripttype_childscript, 'setNumberOfBlocks',
            [blocks], [mass, blockLength, frictionCube, frictionCup], [typeOf],
            emptyBuff, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print(
                'Results: ', retStrings
            )  # display the reply from CoppeliaSim (in this case, the handle of the created dummy)
        else:
            print('Remote function call failed')

    def rotate_cup(self):
        ''' 
        Function to rotate cup
        '''
        errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.source_cup_handle, self.speed,
                                                sim.simx_opmode_oneshot)
        returnCode, self.joint_position = sim.simxGetJointPosition(self.clientID, self.source_cup_handle,
                                                        sim.simx_opmode_buffer)

    def move_cup(self, action):
        ''' 
        Function to move the pouring cup laterally during the rotation. 
        '''
        global low, high
        resolution = 0.001
        move_x = resolution * action
        movement = self.source_cup_position[0] + move_x
        if self.center_position + low < movement < self.center_position + high:
            self.source_cup_position[0] = movement
            returnCode = sim.simxSetObjectPosition(self.clientID, self.source_cup_handle, -1,
                                                self.source_cup_position,
                                                sim.simx_opmode_blocking)

    def discretize_state(self, obs):
        '''
        Bins each dimension of the state space into predefined bins within the initialized bounds.
        Allows the Q-table to be smaller while still accurately representing each state.
        '''
        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(self.lower_bounds[i]))
                / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.bins[i] - 1) * scaling))
            new_obs = min(self.bins[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def normalize(self, val, min_val, max_val):
        '''
        norm_i = (x_i - min(x)) / max(x) - min(x)) * Q
        Normalize values between 0 and Q = 100
        '''
        norm_val = (val - min_val) / (max_val - min_val) * 100
        return norm_val

    def update_q(self, state, action, new_state, reward):
        '''
        Update the Q-table using Bellman equation
        '''
        # Add 2 to action variable to map correctly to Q-table indices
        action += 2
        # Update Q-table for Q(s,a)
        self.q_table[state][action] = \
                self.q_table[state][action] * (1 - self.learning_rate) + \
                self.learning_rate * (reward + self.discount * np.max(self.q_table[new_state]))

        '''self.q_table[state][action] += (self.learning_rate *
                    (reward
                    + self.discount * np.max(self.q_table[new_state])
                    - self.q_table[state][action]))'''
    
    def pick_action(self, state):
        '''
        Exploration-exploitation trade-off. Returns an action based on epsilon-greedy algorithm.
        '''
        # If exploration is picked, select a random action from the current state's row in the q-table
        if (np.random.random() < self.epsilon):
            # Select the largest Q-value
            return self.action_space.sample() - 2
        # If exploitation is picked, select action where max Q-value exists within state's row in the q-table
        else:
            return np.argmax(self.q_table[state]) - 2

    def get_learning_rate(self, t):
        '''
        Get learning rate, which declines after each episode.
        '''
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))
    
    def get_epsilon(self, t):
        '''
        Get epsilon value, which declines after each episode.
        '''
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def wait_(self):
        for _ in range(60):
            self.triggerSim()