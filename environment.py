from gym import Env
from gym.spaces import Discrete, Box
from keras.models import Sequential 
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

import sys
import math
import random
import os.path
import numpy as np
sys.path.append('MacAPI')
import sim


np.set_printoptions(threshold=sys.maxsize)

# Max movement along X
low, high = -0.05, 0.05

def get_distance_3d(a, b):
    a_x, a_y, a_z = a[0], a[1], a[2]
    b_x, b_y, b_z = b[0], b[1], b[2]

    # Distance between source cup and receive cup
    return math.sqrt((a_x - b_x)**2 + (a_y - b_y)**2 + (a_z - b_z)**2)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # 7 -> source_x, and x,y,z for each cube
        self.action_size = action_size # 5 -> [-2, -1, 0, 1, 2]
        #100 for source_x, 100 for x,y,z of each cube
        self.bins = (100, 100, 100, 100, 100, 100, 100)
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

        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.num_episodes = 300
        self.batch_size = 32

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


    def train(self):
        #rewards_all_episodes = []
        #rewards_filename = "rewards_history.txt"

        for e in range(self.num_episodes):
            print(f"Episode {e+1}:")

            # Set rotation velocity randomly
            rng = np.random.default_rng()
            velReal = self.rotation_velocity(rng)

            # Start simulation, process objects/handles
            self.start_simulation()

            # Set initial position of the source cup and initialize state
            #state = self.discretize_state(self.reset(rng))
            state = self.reset(rng)
            state = np.reshape(state, [1, self.state_size])

            #self.learning_rate = self.get_learning_rate(e)
            #self.epsilon = self.get_epsilon(e)
            done = False
            # Keep track of rewards for each episode, to be stored in a list
            #rewards_current_episode = 0

            for j in range(velReal.shape[0]):
                self.step_chores()
                # Initialize the speed of the source cup at this frame
                self.speed = velReal[j]

                # Pick next action, greedy epsilon
                action = self.act(state)

                # Take next action
                next_state, reward, done, _ = self.step(action)

                # Normalize speed value for q_table
                #self.speed = self.normalize(velReal[j], self.velReal_low, self.velReal_high)
                # Calculate new state, continuous -> discrete
                #new_state = self.discretize_state(obs)
            
                # Update Q-table for Q(s,a)
                #print(f"State: {state}, Reward: {reward}, Action: {action}")
                #self.update_q(state, action, new_state, reward)

                next_state = np.reshape(next_state, [1, self.state_size]) # Reshape state for Keras

                self.remember(state, action, reward, next_state, done)

                # Update state variable
                state = next_state

                # Keep track of rewards for current episode
                #rewards_current_episode += reward
                
                # Break if cup goes back to vertical position
                if done:
                    print("episode: {}/{}, epsilon: {:.2}".format(e, self.num_episodes-1, self.epsilon))
                    break
            #end for

            # Stop simulation
            self.stop_simulation()

            if len(self.memory) > self.batch_size:
                self.train_batch()

            if e % 50 == 0:
                self.save("weights_" + "{:04d}".format(e) + ".hdf5")

            # Append current episode's reward to total rewards list for later
            #rewards_all_episodes.append(rewards_current_episode)

        # Append this episodes rewards to a .txt file
        #with open(rewards_filename, 'wb') as f:
        #    np.savetxt(f, rewards_all_episodes)
        #print(f"Saved rewards history to {rewards_filename}")

    def run(self):
        # Load model weights from training
        self.model.load_weights("weights_0250.hdf5")

        for e in range(1):
            print(f"Episode {e+1}:")

            # Set rotation velocity randomly
            rng = np.random.default_rng()
            velReal = self.rotation_velocity(rng)

            # Start simulation, process objects/handles
            self.start_simulation()

            # Set initial position of the source cup and initialize state
            #state = self.discretize_state(self.reset(rng))
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

                # Normalize speed value for q_table
                #self.speed = self.normalize(velReal[j], self.velReal_low, self.velReal_high)
                # Calculate new state, continuous -> discrete
                #new_state = self.discretize_state(obs)
            
                #print(f"State: {state}, Reward: {reward}, Action: {action}")

                next_state = np.reshape(next_state, [1, self.state_size]) # Reshape state for Keras

                #self.remember(state, action, reward, next_state, done)

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
        # Calculate reward as the negative distance between the source up and the receiving cup
        reward_ = -get_distance_3d(self.receive_cup_position, self.source_cup_position)
        min_d = -get_distance_3d(self.receive_cup_position, [-0.8500, -0.1555, 0.7248])
        if self.center_position > -0.8500:
            max_d = -get_distance_3d(self.receive_cup_position, 
                                    [self.center_position+high, -0.1555, 0.7248])
        else:
            max_d = -get_distance_3d(self.receive_cup_position, 
                                    [self.center_position+low, -0.1555, 0.7248])

        if max_d > reward_ > min_d:
            print(f"{min_d}, {max_d}, {reward_}")
        #print(self.normalize(reward_, min_d, max_d, 100))
        flag = 0
        for cube in self.cubes_positions:
            # If cubes are within x bounds of receiving cup rim
            # and above the table
            if (-0.88 < cube[0] < -0.82) and (0.30 < cube[2] < 0.80):
                flag += 1
                
        # Both cubes are lined up along x-axis with receiving cup
        if flag == 2:
            reward = 1000 - self.normalize(reward_, min_d, max_d, 100)
            print(f"reward HIT: {reward}")
        else: 
            reward = 0 - self.normalize(reward_, min_d, max_d, 100)
            print(f"reward MISS: {reward}")

        # Rotate cup based on speed value
        self.rotate_cup()
        # Move cup laterally based on selected action (minus 2 for indexing)
        self.move_cup(action-2)

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

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def save(self, name):
        self.model.save_weights(name)

    def train_batch(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward #if done
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        # Decay learning rate at end of episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print(f"Selecting action based on values: {act_values[0]}\nAction selected: {np.argmax(act_values[0]) - 2}")
        return np.argmax(act_values[0])
    
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