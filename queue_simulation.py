import numpy as np
import random
import matplotlib.pyplot as plt 

# Set a fixed seed for random number generation for reproducibility.
random.seed(1707406865)

# In the simulation, we used a factory software design pattern.
# This allows our code to scale for multiple queues - not just for two!

class Queue:
    # Initializes the Queue with given parameters.
    def __init__(self, service_rate, queue_participants, service_distribution, arrival_distribution, arrival_rate = None):
        
        # Number of participants in the queue.
        self.queue_participants = queue_participants
        
        # Set up interarrival times based on the arrival distribution. The second block of code
        # is triggered for the second queue in the system, where the arrival, waiting, and service time
        # in the first queue summed gives the interarrival times of the second queue!
        if isinstance(arrival_distribution, str) and arrival_distribution == 'markov':
            self.interarrival_times = [random.expovariate(arrival_rate) for _ in range(self.queue_participants)]
        elif isinstance(arrival_distribution, list):
            self.interarrival_times = arrival_distribution
              
        # Set up service times based on the service distribution.          
        if service_distribution == 'markov':
            self.service_times = [random.expovariate(service_rate) for _ in range(self.queue_participants)]
        elif service_distribution == 'deterministic':
            self.service_times = [service_rate for _ in range(self.queue_participants)]
            
    def time_in_queue(self, second_queue = False):
        
        # Calculate the total time spent in the queue by each participant. 
        # Converts interarrival times to arrival times. Again, the second block is triggered
        # for the second queue of the system!
        if second_queue is False:
            arrival_times = list(np.cumsum(self.interarrival_times))
        else:
            arrival_times = self.interarrival_times
        
        # Initialises the output list, and the time for the first queue participant
        output = [None for _ in range(self.queue_participants)]
        output[0] = arrival_times[0] + self.service_times[0]
        
        # Simulate the queue for participants by recursively comparing arrival times!
        for i, item in enumerate(arrival_times):
            if i < len(arrival_times) - 1:
                if arrival_times[i+1] < output[i]:
                    output[i+1] = output[i] + self.service_times[i+1]
                elif arrival_times[i+1] > output[i]:
                    output[i+1] = arrival_times[i+1] + self.service_times[i+1]
        
        # Calculates total time in queue        
        total_time_in_queue = [output[_] - arrival_times[_] for _ in range(self.queue_participants)]
        
        # This flag is necessary as the arrival_times with the service times of the first queue
        # comprise of the arrival_times of the second queue!
        if second_queue is False:
            return total_time_in_queue, output
        
        else:
            return total_time_in_queue

class QueueSystem:
    # Initializes a system with one or more queues. By design, this should
    # be easily modifiable to scale for more than two queues!
    def __init__(self, first_queue: Queue):
        self.first_queue = first_queue
        
    def time_in_system(self, service_rate, service_distribution):
        # Calculate the total time spent in the system (across all queues).
        total_time_first_queue, arrival_time_second_queue = self.first_queue.time_in_queue(second_queue = False)
        second_queue = Queue(service_rate, self.first_queue.queue_participants, service_distribution, arrival_time_second_queue, arrival_rate = None)
        total_time_second_queue = second_queue.time_in_queue(second_queue = True)
        
        # Sum time in first and second queues.
        total_time_in_system = total_time_first_queue + total_time_second_queue
        return total_time_in_system
        
    @staticmethod
    def create_histogram(time_in_system_one, time_in_system_two):
        
        # A bin width of 0.2
        # Define bin width for histograms.
        bin_width = 0.2
        bin_one = np.arange(min(time_in_system_one), max(time_in_system_one) + bin_width, bin_width)
        bin_two = np.arange(min(time_in_system_two), max(time_in_system_two) + bin_width, bin_width)
        
        # Create a histogram of the times
        hist_one, bin_one = np.histogram(time_in_system_one, bins=bin_one, density=True)
        bin_centers_one = (bin_one[:-1] + bin_one[1:]) / 2
        hist_two, bin_two = np.histogram(time_in_system_two, bins=bin_two, density=True)
        bin_centers_two = (bin_two[:-1] + bin_two[1:]) / 2

        # Plot the histogram
        plt.bar(bin_centers_one, hist_one, width=bin_centers_one[1] - bin_centers_one[0], color='blue', alpha=0.7, label=f'Simulated (mu1=2, mu2=1)')
        plt.bar(bin_centers_two, hist_two, width=bin_centers_two[1] - bin_centers_two[0], color='red', alpha=0.7, label=f'Simulated (mu1=1, mu2=2)')

        # Label the axes and show the plot
        plt.xlabel('Time in system')
        plt.ylabel('Probability density')
        plt.title('Distribution of Time Spent in Two Queue in Series')
        plt.legend()
        plt.show()
        
queue_series_one = Queue(service_rate = 2, queue_participants = 1000, arrival_distribution = 'markov', service_distribution = 'markov', arrival_rate = 1)
queue_system_one = QueueSystem(queue_series_one)
total_time_one = queue_system_one.time_in_system(1, 'deterministic')

queue_series_two = Queue(service_rate = 1, queue_participants = 1000, arrival_distribution = 'markov', service_distribution = 'deterministic', arrival_rate = 1)
queue_system_two = QueueSystem(queue_series_two)
total_time_two = queue_system_two.time_in_system(2, 'markov')

QueueSystem.create_histogram(total_time_one, total_time_two)