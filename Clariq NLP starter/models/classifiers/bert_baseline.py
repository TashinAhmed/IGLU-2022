
import numpy as np
import torch
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification

class BERTClassifier:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("models/classifiers/bert_baseline/saved_tokenizer")
        self.max_seq_length = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = RobertaForSequenceClassification.from_pretrained("models/classifiers/bert_baseline/saved_model")
        self.model.to(self.device)

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

        with torch.no_grad():
            encoded_dict = self.tokenizer.encode_plus(
                                instruction.lower(),
                                add_special_tokens = True, 
                                max_length = self.max_seq_length,           
                                pad_to_max_length = True,
                                truncation=True, 
                                return_attention_mask = True,   
                                return_tensors = 'pt',     
                        )
            inputs = encoded_dict['input_ids'].to(self.device)
            attention_mask = encoded_dict['attention_mask'].to(self.device)
            results = self.model(inputs, attention_mask=attention_mask)
            
        return np.argmax(results.logits.cpu().numpy())
