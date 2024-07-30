import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticModel(nn.Module):
    def __init__(self, num_tasks, num_workers, task_feature_dim, worker_feature_dim):
        super(ActorCriticModel, self).__init__()
        # Convolutional layer for task feature input
        self.task_conv1 = nn.Conv1d(task_feature_dim, 128, kernel_size=1)

        # Convolutional layer for worker feature input
        self.worker_conv1 = nn.Conv1d(worker_feature_dim, 128, kernel_size=1)

        # Dense layers for processing current time and last performed task inputs
        self.dense1_time = nn.Linear(1, 128)
        self.dense2_time = nn.Linear(128, 128)

        self.dense1_last_task = nn.Linear(1, 128)
        self.dense2_last_task = nn.Linear(128, 128)

        # Final dense layer for the actor network
        self.actor_dense = nn.Linear(num_tasks * 128 + num_workers * 128 + 256, num_workers)

        # Final dense layer for the critic network
        self.critic_dense = nn.Linear(num_tasks * 128 + num_workers * 128 + 256, 1)

    def forward(self, task_features, worker_features, time_input, last_task_input):
        # Task feature input processing
        task_features = task_features.permute(0, 2, 1)  # Change shape to (batch_size, task_feature_dim, num_tasks)
        task_x = F.relu(self.task_conv1(task_features))
        task_x = task_x.reshape(task_x.size(0),-1) #task_x.view(task_x.size(0), -1)  # Flatten to (batch_size, num_tasks * 128)

        # Worker feature input processing
        worker_features = worker_features.permute(0, 2, 1)  # Change shape to (batch_size, worker_feature_dim, num_workers)
        worker_x = F.relu(self.worker_conv1(worker_features))
        worker_x = task_x.reshape(task_x.size(0),-1) #worker_x.view(worker_x.size(0), -1)  # Flatten to (batch_size, num_workers * 128)

        # Time input processing
        time_x = F.relu(self.dense1_time(time_input))
        time_x = F.relu(self.dense2_time(time_x))

        # Last task input processing
        last_task_x = F.relu(self.dense1_last_task(last_task_input))
        last_task_x = F.relu(self.dense2_last_task(last_task_x))

        #shapeptint
        # print('taskx', task_x.size())
        # print('last_task_x', last_task_x.size())
        # print('time_x', time_x.size())
        # print('worker_x', worker_x.size())
        # Concatenate all processed inputs
        concat = torch.cat((task_x, worker_x, time_x, last_task_x), dim=1)

        # Actor network
        actor_output = self.actor_dense(concat)
        # actor_output = F.softmax(actor_output, dim=1)  # Apply masking

        # Critic network
        critic_output = self.critic_dense(concat)

        return actor_output, critic_output