import numpy as np

class RandomClassifier:
    def __init__(self):
        pass

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def clarification_required(self, instruction, gridworld_state):
        """
        Implements classifier for given instuction - whether a clarifying question is required or not
        Inputs:
            instruction - Single instruction string

            gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                              NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

        Outputs:
            0 or 1 - 0 if clarification is not required, 1 if clarification is required 

        """

        return np.random.choice([0, 1])