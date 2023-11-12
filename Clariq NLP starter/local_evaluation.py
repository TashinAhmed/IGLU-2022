import pandas as pd
import os
import numpy as np
import json
from sklearn.metrics import f1_score
from models.user_config import UserRanker, UserClassifer
from tqdm.auto import tqdm
import warnings

def check_data(datafolder):
    """
    Checks if the data is downloaded and placed correctly
    Checks for state data else skips it with warning
    """
    datafile = os.path.join(datafolder, 'clarifying_questions_train.csv')
    if not os.path.isfile(datafile):
        raise NameError("Please download the public data from \n https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge/problems/neurips-2022-iglu-challenge-nlp-task/dataset_files")
    
    if not os.path.isfile(os.path.join(datafolder, 'question_bank.json')):
        question_bank = {"question_bank": pd.read_csv(datafile).ClarifyingQuestion.dropna().tolist()}
        with open(os.path.join(datafolder, 'question_bank.json'), 'w') as fp:
            json.dump(question_bank, fp)

    if not os.path.isdir(os.path.join(datafolder, 'initial_world_states')):
        warnings.warn("Please unzip the task states zip and place the initial_world_states directory in the public_data folder",
                       UserWarning)
        warnings.warn("Skipping state usage", UserWarning)
        return False
    else:
        return True 
    


def read_json_file(fname):
    with open(fname, 'r') as fp:
        d = json.load(fp)
    return d

def get_gridworld_state(index, instructions_df, datafolder, states_available):
    if not states_available:
        return None
    state_file_path = instructions_df.loc[index].InitializedWorldPath
    state_file_path = os.path.join(datafolder, state_file_path)
    state = read_json_file(state_file_path)
    for drop_key in ['gameId', 'stepId', 'tape', 'clarification_question']:
        state.pop(drop_key)
    return state

def run_classification(classifier, instructions_df, LocalEvalConfig, states_available):
    predictions = {}
    for index, row in tqdm(instructions_df.iterrows(), 
                           total=len(instructions_df),
                           desc='Running classifier'):
        gridworld_state = get_gridworld_state(index=index, 
                                              instructions_df=instructions_df, 
                                              datafolder=LocalEvalConfig.DATA_FOLDER, 
                                              states_available=states_available)

        res = classifier.clarification_required(row.InputInstruction, gridworld_state)
        assert res in [0, 1], "Result of classfier should be 0 or 1"
        predictions[row.InputInstructionWithGameID] = int(res)
        
    with open(LocalEvalConfig.CLASSIFIER_RESULTS_FILE, 'w') as fp:
        json.dump(predictions, fp)



def run_ranking(ranker, instructions_df, LocalEvalConfig, states_available):
    predictions = {}
    instructions_unclear = instructions_df.dropna(subset=['ClarifyingQuestion'], inplace=False)
    for index, row in tqdm(instructions_unclear.iterrows(), 
                           total=len(instructions_unclear),
                           desc='Running ranker'):

        gridworld_state = get_gridworld_state(index=index, 
                                              instructions_df=instructions_unclear, 
                                              datafolder=LocalEvalConfig.DATA_FOLDER, 
                                              states_available=states_available)

        question_bank_path = os.path.join(LocalEvalConfig.DATA_FOLDER, 'question_bank.json')
        question_bank = read_json_file(question_bank_path)["question_bank"]

        res = ranker.rank_questions(row.InputInstruction, gridworld_state, question_bank)

        assert isinstance(res, list) or isinstance(res, tuple), "Output of ranker must be a list/tuple of strings"
        predictions[row.InputInstructionWithGameID] = list(res)
    
    with open(LocalEvalConfig.RANKER_RESULTS_FILE, 'w') as fp:
        json.dump(predictions, fp)


def evaluate(LocalEvalConfig):
    """
    Runs the local evaluation in same way as the aicrowd evaluator
    Both ranker and classifer will be run independently in this code
    Final evaluation code is the same as the evaluator
    """

    states_available = check_data(LocalEvalConfig.DATA_FOLDER)

    instructions_df = pd.read_csv(os.path.join(LocalEvalConfig.DATA_FOLDER, 'clarifying_questions_train.csv'))
    instructions_df['InputInstructionWithGameID'] = instructions_df.InputInstruction + instructions_df.GameId
    
    # Run classfier
    classifier = UserClassifer()
    run_classification(classifier, instructions_df, LocalEvalConfig, states_available)

    # Run ranker
    ranker = UserRanker()
    run_ranking(ranker, instructions_df, LocalEvalConfig, states_available)

    # Load prediction files
    classifer_preds = read_json_file(LocalEvalConfig.CLASSIFIER_RESULTS_FILE)
    ranker_preds = read_json_file(LocalEvalConfig.RANKER_RESULTS_FILE)

    # Get classfication score
    classifier_gt = pd.Series(instructions_df.IsInstructionClear.values, 
                              index=instructions_df.InputInstructionWithGameID).to_dict()
    cpreds, cgt = [], []
    for instructionWithGameID, instruction_is_clear in classifier_gt.items():
        cgt.append(int(instruction_is_clear.lower() == 'no'))
        pred = classifer_preds.get(instructionWithGameID, None)
        if pred is not None:
            cpreds.append(pred)
        else:
            warnings.warn(f"No prediction for instruction + game id {instructionWithGameID}")
            # if any instruction is not predicted, default value will be taken as 1
            cpred.append(0)

    clariq_f1_score = f1_score(y_true=cgt, y_pred=cpreds, average='macro')

    # Get ranker score
    unclear_rows = instructions_df.dropna(subset=['ClarifyingQuestion'], inplace=False)
    ranker_gt = pd.Series(unclear_rows.ClarifyingQuestion.values, index=unclear_rows.InputInstructionWithGameID).to_dict()
    inverse_rank_scores = []
    for instructionWithGameID, clarifying_question  in ranker_gt.items():
        qpreds = ranker_preds[instructionWithGameID]
        if clarifying_question in qpreds:
            inverse_rank_scores.append(1/(qpreds.index(clarifying_question) + 1))
        else:
            inverse_rank_scores.append(0.0)
        
    clariq_mrr = np.mean(inverse_rank_scores)

    f1_score_bins = [0.9, 0.85, 0.75, 0.65, 0.5, 0.35]
    binned_f1 = max([bin_floor * (clariq_f1_score > bin_floor) for bin_floor in f1_score_bins])

    print("===================== Final scores =======================")
    print("Binned F1 Score", binned_f1)
    print("MRR for Ranker", clariq_mrr)
    print("F1 Score for Classifier", clariq_f1_score)



if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        CLASSIFIER_RESULTS_FILE = './local-eval-classifier-results.json'
        RANKER_RESULTS_FILE = './local-eval-ranker-results.json'
        DATA_FOLDER = './public_data'
    
    evaluate(LocalEvalConfig)
