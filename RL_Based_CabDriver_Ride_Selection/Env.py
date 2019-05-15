# Import routines

import numpy as np
import itertools as iter_tool
import math
import random
from datetime import datetime

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

Time_matrix=np.load("TM.npy") # Time Matrix used to calculate the commute time


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = self.init_action_space()
        self.state_space =self.init_state_space()
        self.state_init = self.init_get_state()
        # Start the first round
        self.reset()
        
        self.debug = True

    ## Encoding state (or state-action) for NN input
    
    
    def debug_print(self,message):
        if self.debug:
            print("%s : DEBUG : %s" %(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),str(message)))
        return
    
    def init_get_state(self):
        city_list=[city for city in range(1,m+1)]
        driver_current_location=np.random.choice(city_list) #selcting 1 option from m given number of city
        
        hour_list=[hour for hour in range(0,t)] #number of hours, ranges from 0 .... t-1
        hour_of_the_day=np.random.choice(hour_list) # Selecting 1 option from t-1 given hours
        
        day_list=[day for day in range(0,d)] # number of days, ranges from 0 ... d-1
        day_of_the_week=np.random.choice(day_list)
        
        #We must return the state as tuple so that it can be used as key in Dictionary (tuple is immutable so dictionary works)
        return (driver_current_location,hour_of_the_day,day_of_the_week)
    
    def init_action_space(self):
        """action space is of form (pickup,drop) tuples where pickup != drop 
        and 1<p<m and 1<q<m
        also if the driver dont want to pickup and drop then (0,0) tuple is needed to be appended to main list
        for m cities the action_space should have length (m-1)*m+1"""
        
        action_space = []
        
        for start in range(1,m+1):
            for end in range(1,m+1):
                if start != end:
                      action_space.append((start,end))       
        
        # appending (0,0) tuple for no pickup no drop condition
        action_space.append((0,0))
      
        #Returning action space which will be of format [(0,0),(1,2),(1,3)....(5,4)..(5,2).(5,1)] total (m-1)*m+1
        return action_space
    
    def init_state_space(self):
        
        city_list=[city for city in range(1,m+1)] #getting list of location of the CAB
        hour_list=[hour for hour in range(0,t)] #getting list of hour of the day
        day_list=[day for day in range(0,d)] #getting list of day of the week (7 day max)
        
        state_space=list(iter_tool.product(city_list,hour_list,day_list))
        
        return state_space
    
    # Use this function if you are using architecture-1
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
         # format of state is tuple (driver_current_location,hour_of_the_day,day_of_the_week)
           #for state tuple
        
        driver_location=np.zeros(m)
        hour_of_day=np.zeros(t)
        day_of_week=np.zeros(d)
        
        #the reason index is used as state[0]-1 is because location(city or m) varies from 1-5 so index varies from 0-4
        driver_location[state[0]-1]=1
        hour_of_day[state[1]]=1
        day_of_week[state[2]]=1
        
         #Creating combined list by extension
        state_encod = np.hstack((driver_location,hour_of_day,day_of_week)).reshape(1,m+t+d)
        
        return state_encod
    
    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format.
         Hint: The vector is of size m + t + d + m + m."""
         # format of state is tuple (driver_current_location,hour_of_the_day,day_of_the_week)
         #format of action is tuple (from_location,to_location)
         #initializing all one hot vectors as zeros
         #for state tuple
        driver_location=np.zeros(m)
        hour_of_day=np.zeros(t)
        day_of_week=np.zeros(d)
         
        #for action tuple below
        # There can be (0,0) action as well, so we should have (m+1)
        from_location=np.zeros(m+1)
        to_location=np.zeros(m+1)
         
        #the reason index is used as state[0]-1 is because location(city or m) varies from 1-5 so index varies from 0-4
        driver_location[state[0]-1]=1
        hour_of_day[state[1]]=1
        day_of_week[state[2]]=1
         
        #format of action is tuple (from_location,to_location)"""
         
        # in this action part of the encoded vector would look something like this #
        # from_location = [X - X - X - X - X - X]
        #                  ^
        #                  |
        #              action(0,0) -> Others are for from_location (1 or 2 or 3 or 4 or 5)
        
        # to_location = [X - X - X - X - X - X]
        #                ^
        #                |
        #         action(0,0) -> Others are for to_location (1 or 2 or 3 or 4 or 5)
        
        if action == (0,0):
            from_location[0] = 1
            to_location[0] = 1
        else:
            from_location[action[0]]=1
            to_location[action[1]]=1
         
         #Creating combined list by extension
        state_encod = np.hstack((driver_location,hour_of_day,day_of_week,from_location,to_location)).reshape(1,m+t+d+(m+1)+(m+1))
         
        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        
        n_requests = 0
        
        if location == 1:
            n_requests = np.random.poisson(2)
        elif location==2:
            n_requests=np.random.poisson(12)
        elif location==3:
            n_requests=np.random.poisson(4)
        elif location==4:    
            n_requests=np.random.poisson(7)
        elif location==5:
            n_requests=np.random.poisson(8)

        if n_requests > 15:
            n_requests = 15
        
        
        self.debug_print("def requests: location = %d, n_requests (Poisson) = %d" %(location,n_requests))
        
        # We should be returning the index's of self.action_space which are possible at this momemnt
        # Remember, we added (0,0) at the last index = 20
        
        action_index = random.sample(range(0,(m-1)*m),n_requests)
        action_index.append(20)
        
        self.debug_print("def requests: action_index chosen : %s" %(str(action_index)))

        possible_action = [self.action_space[x] for x in action_index]

        return action_index,possible_action   
    
    def execute_ride_action(self,state, action):
        """Below code returns Time of the day after taking action from any state.
        1.It also check if the time >24 hours in (0-23) hours range then it returns 24-calculated_hours.
        2.It check if he have to travel to pickup location from driver's current location"""       
    
        given_state=state
        
        current_location=given_state[0]
        current_hour_of_the_day=given_state[1]
        day_of_the_week=given_state[2]
        
        location_from=action[0]
        location_to=action[1]
        
        if action == (0,0):
            # Driver chooses not to take the ride, hence time increased by an hour
            # ride time = 0
            # cost time = 1
            # Reward will be calculated by calculate_reward function
            
            self.debug_print("Action : %s" %(str(action)))
            
            next_location = current_location
            
            ride_time = 0.0
            cost_time = 1.0
            
            next_time = current_hour_of_the_day + 1
            next_day = day_of_the_week
            
            if next_time >= 24:
                next_time = int(next_time - 24)
                if day_of_the_week == 6:
                    next_day = 0
                else:
                    next_day = next_day + 1
            
            reward = self.calculate_reward(ride_time,cost_time)
            
            next_state = (next_location,next_time,next_day)
            
            self.debug_print("next_state : %s, next_time : %d, next_day : %d, ride_time : %f, cost_time : %f, reward : %f" %(str(next_state),next_time,next_day,ride_time,cost_time,reward))
            
            return next_state,reward,cost_time
            
        elif location_from == current_location:
            # Driver chooses to take the ride (he is at same location as the request location)
            # ride time = to be calculated
            # cost time to be calculated
            # reward will be calculated by calculate_reward function
            self.debug_print("Action : %s, driver at customer request location" %(str(action)))
            next_location = location_to
            
            ride_time = Time_matrix[int(location_from-1)]\
                                   [int(location_to-1)]\
                                   [int(current_hour_of_the_day)]\
                                   [int(day_of_the_week)]
                                   
            cost_time = ride_time
            
            next_time = int(current_hour_of_the_day + ride_time)
            next_day = day_of_the_week
            
            if next_time >= 24:
                next_time = int(next_time - 24)
                if day_of_the_week == 6:
                    next_day = 0
                else:
                    next_day = next_day + 1
            
            reward = self.calculate_reward(ride_time,cost_time)
            
            next_state = (next_location,next_time,next_day)
            
            self.debug_print("next_state : %s, next_time : %d, next_day : %d, ride_time : %f, cost_time : %f, reward : %f" %(str(next_state),next_time,next_day,ride_time,cost_time,reward))
            
            return next_state,reward,cost_time
        
        else :
            # Driver chooses to take the ride (he is NOT at the same location as the request location)
            # ride time = to be calculated
            # cost time = to be calculated
            
            self.debug_print("Action : %s, driver NOT at customer request location" %(str(action)))
            
            next_location = location_to
            
            time_to_reach_customer_location = Time_matrix[int(current_location-1)]\
                                                         [int(location_from-1)]\
                                                         [int(current_hour_of_the_day)]\
                                                         [int(day_of_the_week)]
            
            time_at_customer_location = int(current_hour_of_the_day + time_to_reach_customer_location)
            day_at_customer_location = day_of_the_week
            
            if time_at_customer_location >= 24:
                time_at_customer_location = int(time_at_customer_location - 24)
                if day_of_the_week == 6:
                    day_at_customer_location = 0
                else:
                    day_at_customer_location = day_of_the_week + 1
                    
            self.debug_print("After going to customer location, time : %d, day : %d" %(time_at_customer_location,day_at_customer_location))
            
            time_to_reach_final_location = Time_matrix[int(location_from-1)]\
                                                      [int(location_to-1)]\
                                                      [int(time_at_customer_location)]\
                                                      [int(day_at_customer_location)]
            
            time_at_final_location = int(time_at_customer_location + time_to_reach_final_location)
            day_at_final_location = day_at_customer_location
            
            if time_at_final_location >= 24:
                time_at_final_location = int(time_at_final_location - 24)
                if day_at_customer_location == 6:
                    day_at_final_location = 0
                else:
                    day_at_final_location = day_at_customer_location + 1
                    
            self.debug_print("After going to final location, time : %d, day : %d" %(time_at_final_location,day_at_final_location))
            
            next_time = time_at_final_location
            next_day = day_at_final_location
            
            ride_time = time_to_reach_final_location
            cost_time = time_to_reach_customer_location + time_to_reach_final_location
            
            reward = self.calculate_reward(ride_time,cost_time)
            
            next_state = (next_location,next_time,next_day)
            
            self.debug_print("next_state : %s, next_time : %d, next_day : %d, ride_time : %f, cost_time : %f, reward : %f" %(str(next_state),next_time,next_day,ride_time,cost_time,reward))
            
            return next_state,reward,cost_time
        
    def calculate_reward(self, ride_time, cost_time):
        return (R*ride_time) - (C*cost_time)
    
    """This Step function returns the next state and reward"""
    def step(self,state,action):
        next_state, reward, cost_time = self.execute_ride_action(state,action)
        return next_state, reward, cost_time

    def reset(self):
        return self.action_space, self.state_space, self.state_init
