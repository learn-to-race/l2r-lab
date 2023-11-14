# Metrics and Reward Functions
- Average Speed (kph): This metric calculates the average speed of an agent in an episode before it terminates. Higher values indicate better performance.

- Movement Smoothness: This metric is defined as the log of the dimensionless 'jerk' based on accelerometer data collected by the agent. Higher values indicate smoother movements, which are better.

- Displacement Error: This metric measures the Euclidean displacement from the track centerline, in meters. Lower values indicate that the agent is straying less from the centerline, which is better.

- Distance Covered: This metric calculates the distance travelled by the agent in an episode before it terminates. Higher values indicate better performance.

- Time taken: This metric measures the time taken by an agent to complete an episode. This can be used in conjunction with other metrics since low time taken could mean that the agent has learned to go fast or that the agent is crashing very quickly.

# Current Architecture - IMPALA
- One learner that learns, and multiple workers that collect experience by simulating environments.
    - Pros:
    - Cons:
    - Won't this make the learner a bottleneck?
        - We can have multiple learners work together to process the workers' feedback
        - Asynchronous learning - learner updates models as soon as they receive new experience (instability during training)
    - Should each worker collect from different environment?
        - Generally yes, for wider variety of scenarios
    - What's importance sampling?
        - Estimate the expected value of the return function with respect to the true environment distribution, using data collected from a different distribution
        - Re-weighting the samples collected from the behavior policy to give more weight to the samples that are more relevant to the target policy
        - Allows us to use the data collected by different workers, which may be generated from different behavior policies or different parts of the environment, to improve the mode
        - [For L2R] - For example, suppose that the goal of the learning process is to improve the car's ability to take turns at high speeds. In that case, experiences where the car successfully navigates a difficult turn at high speed might be more valuable than experiences where the car drives in a straight line. By using importance sampling, we can give more weight to the experiences that are more valuable for the learning process, and reduce the weight of experiences that are less valuable
        - How to implement - (1) A way to calculate the probability of each action taken by the actor network, given a state. This is called the "policy" or "behavior" policy. (2) A way to calculate the probability of each action taken by the target network, given the same state. This is called the "target" policy. (3) The ratio of the two probabilities, which is used to weight the importance of each sample. (4) The actual sample of experience, which includes the state, action, reward, and next state.
- What could have led to a decreased performance?
    - Communication overhead - when using distributed reinforcement learning methods like IMPALA, there is additional overhead associated with communication between the learner and actors. This can include the time it takes to send data back and forth between machines, as well as the time it takes for the learner to process the collected data. If the communication overhead is too high, it can slow down the learning process and lead to a decrease in performance.
    - Actor sampling - the actors are responsible for collecting data from the environment and sending it to the learner. If the actors are not sampling from the environment in a diverse or representative way, it can lead to biased data being collected and a decrease in performance.
    - Model synchronization - the model parameters need to be synchronized between the actors and the learner. If there are issues with model synchronization, it can lead to a decrease in performance.
    
