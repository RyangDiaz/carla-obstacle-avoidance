
import torch.nn as nn
import torch


class Join(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(Join, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'mode' not in params:
            raise ValueError(" Missing the mode parameter ")
        if 'after_process' not in params:
            raise ValueError(" Missing the after_process parameter ")

        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization

        self.after_process = params['after_process']
        self.mode = params['mode']

    def forward(self, x, m, b):
        # get only the speeds from measurement labels
        if self.mode == 'cat':
            j = torch.cat((x, m, b), 1)

        else:
            raise ValueError("Mode to join networks not found")

        return self.after_process(j)


    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .
        """
        coil_logger.add_message('Loading', {
                    "Model": {"Loaded checkpoint: " + str(checkpoint) }

                })
