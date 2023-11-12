![IGLU Banner](https://user-images.githubusercontent.com/660004/179000978-29cf4462-4d2b-4623-8418-157449322fda.png)

# **[NeurIPS 2022 - IGLU Challenge](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)** - Starter Kit
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the IGLU Challenge **Starter kit**! It contains:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

Quick Links:

* [The IGLU Challenge - Competition Page](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)
* [The IGLU Challenge - Slack Workspace](https://join.slack.com/t/igluorg/shared_invite/zt-zzlc1qpy-X6JBgRtwx1w_CBqOV5~jaA&sa=D&sntz=1&usg=AOvVaw33cSaYXeinlMWYC6bGIe33)
* [The IGLU NLP Challenge Overview](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022/iglu-2022-challenge-checklist/Overview-NLP.md)


# Table of Contents
1. [Intro to the NLP Task](#intro-to-the-nlp-task)
2. [Evaluation](#evaluation)
3. [Baselines](#baselines) 
4. [How to test and debug locally](#how-to-test-and-debug-locally)
5. [How to submit](#how-to-submit)
6. [Dataset](#dataset)
7. [Setting up your codebase](#setting-up-your-codebase)
8. [FAQs](#faqs)

# Intro to the NLP Task: Asking Clarifying Questions

This task is about determining **when and what clarifying questions** to ask. Given the instruction from the Architect (e.g., “Help me build a house.”), the Builder needs to decide whether it has sufficient information to carry out that described task or if further clarification is needed. For instance, the Builder might ask “What material should I use to build the house?” or “Where do you want it?”. In this NLP task, we focus on the research question "what to ask to clarify a given instruction" independently from learning to interact with the 3D environment. The original instruction and its clarification can be used as input for the Builder to guide its progress.

<img src="https://user-images.githubusercontent.com/660004/178754025-1966703c-3e99-4e59-ad79-bf7257e3d35b.png" width="720">

*Top: architect's instruction was clear, not clarifying question gets asked. Bottom: 'leftmost' is ambiguous, so the builder asks a clarifying question.*

# Evaluation

Models submitted to the NLP track are going to be evaluated according to both *when to ask* and *what to ask* criteria with a two-step scoring process.

- **When to ask:** This is a binary classification problem: Does the provided instruction require a clarification question? We use the the [macro average F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) to evaluate your classifer. However, we do not believe optimizing this metric too much is in the best use of your time from a research perspective. Hence we quantize the F1 score into the following bins:

  - `0.90 - 1.0` 
  - `0.85 - 0.90`
  - `0.75 - 0.85`
  - `0.65 - 0.75` 
  - `0.50 - 0.65`
  - `0.35 - 0.50`
  - `0.0 - 0.35`

  So if your classifer gets a F1 score of 0.82, the binned F1 score will be 0.75. For a F1 score of 0.93, the binned score will be 0.90 and so on.

- **What to ask:** The second problem evaluates how well your model can rank a list of human-issued clarifying questions for a given ambiguous instruction. Your model will be evaluated on [Mean Reciprocal Rank (MRR)](https://machinelearning.wtf/terms/mean-reciprocal-rank-mrr/), rounded off to ***3 significant digits***.

The leaderboard will be ranked based on the binned F1 score, submissions with the same binned F1 score will be sorted with the MRR.

*Please note above mentioned metrics is subject to be modified after completion of warm-up phase of the competition.*


# Baselines

The official baseline is at the link below.

https://www.aicrowd.com/showcase/baseline-bert-classifier-bm25-ranker

We train a binary classifier with BERT using huggingface transformers. We also provide an example of using BM25 to rank the set of questions.

# How to Test and Debug Locally

The best way to test your models is to run your submission locally.

You can do this naively by simply running  `python local_evaluation.py`. 

# How to Submit

More information on submissions can be found at our [SUBMISSION.md](/docs/submission.md).

#### A high level description of the Challenge Procedure:
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge).
2. **Clone** this repo and start developing your solution.
3. **Train** your models on IGLU, and ensure run.sh will generate rollouts.
4. **Submit** your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com)
for evaluation (full instructions below). The automated evaluation setup
will evaluate the submissions against the IGLU Gridworld environment for a fixed 
number of rollouts to compute and report the metrics on the leaderboard
of the competition.


# Dataset

Download the public dataset for this Task using the link below, you'll need to accept the rules of the competition to access the data.

https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge-nlp-task/dataset_files

The dataset consists of 

  - `clarifying_questions_train.csv` 
  - `question_bank.csv`
  - `initial_world_paths` folder

`clarifying_questions_train.csv` has the following columns:

  * `GameId` - Id of the game session.
  * `InitializedWorldPath` - Path to the file under `initial_world_paths` that contains state of the world intialized to the architect. The architect provides an instruction to build based on this world state. More information to follow on how the world state can be parsed/ visualized. 
  * `InputInstruction` - Instruction provided by the architect.
  * `IsInstructionClear` - Specifies whether the instruction provided by architect is clear or ambiguous. This has been marked by another annotator who is not the architect.
  * `ClarifyingQuestion` - Question asked by annotator upon marking instruction as being ambiguous.
  * `qrel` - Question id (qid) of the relevant clarifying question for the current instruction.
  * `qbank` - List of clarifying question ids that need to be ranked for each unclear instruction. The mapping between clarifying questions and ids is present in the `question_bank.csv`.

*Merged list of ids in the `qrel` and `qbank` columns will give you the list of all qids to be ranked for each ambiguous instruction.*

`question_bank.csv`: This file contains mapping between `qids` mentioned in `qrel` and `qbank` columns of the `clarifying_questions_train.csv` to the bank of clarifying questions issued by annotators.


# Setting Up Your Codebase

AIcrowd provides great flexibility in the details of your submission!  
Find the answers to FAQs about submission structure below, followed by 
the guide for setting up this starter kit and linking it to the AIcrowd 
GitLab.

## FAQs

* How do I submit a model?
  * More information on submissions can be found at our [SUBMISSION.md](/docs/submission.md). In short, you should push you code to the AIcrowd's gitlab with a specific git tag and the evaluation will be triggered automatically.

### How do I specify my dependencies?

We accept submissions with custom runtimes, so you can choose your 
favorite! The configuration files typically include `requirements.txt` 
(pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the [RUNTIME.md](/docs/RUNTIME.md) file.

### What should my code structure look like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:


```
.
├── aicrowd.json           # Set your username, IMPORTANT: set gpu to true if you need it
├── apt.txt                # Linux packages to be installed inside docker image
├── requirements.txt       # Python packages to be installed
├── local_evaluation.py    # Use this to check your agent evaluation flow locally
├── evaluator/             # Contains helper functions for local evaluation
└── models                 # Place your models and related code here
    ├── rankers            # Folder keep all your ranker code
    ├── classifiers        # Folder keep all your classifier code
    └── user_config.py     # IMPORTANT: Add your classifer and ranker name here
```


Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** See [How do I actually make a submission?](#how-do-i-actually-make-a-submission) below for more details.


### How can I get going with an existing baseline?

See [baselines section](#baselines)

### How can I get going with a completely new model?

Train your model as you like, and when you’re ready to submit, implement the inference class in the `models/classifiers` and `models/rankers` folders. Refer to [`models/README.md`](models/README.md) for a detailed explanation.

Once you are ready, test your implementation `python local_evaluation.py`

### How do I actually make a submission?

First you need to fill in you `aicrowd.json`, to give AIcrowd some info so you can be scored.
The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "neurips-2022-the-iglu-challenge-nlp-task",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
  "gpu": true
}
```

The submission is made by adding everything including the model to git,
tagging the submission with a git tag that starts with `submission-`, and 
pushing to AIcrowd's GitLab. The rest is done for you!

More details are available [docs/submission.md](docs/submission.md).

### Are there any hardware or time constraints?

Your submission will need to complete all the **ranking tasks** under **60 minutes** 
and all the **classfying tasks** under **30 minutes**. Make sure you take advantage 
of all the cores by parallelizing your code if needed. 

The machine where the submission will run will have following specifications:
* 1 NVIDIA T4 GPU with 16GB mem
* 8 vCPUs
* 56GB RAM

## Contributors

- Negar Arabzadeh
- Shrestha Mohanty
- [Dipam Chakraborty](https://www.aicrowd.com/participants/dipam)

# 📎 Important links
- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge/problems/neurips-2022-iglu-challenge-nlp-task
- 🗣️ Discussion Forum: https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge/problems/neurips-2022-iglu-challenge-nlp-task/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge/problems/neurips-2022-iglu-challenge-nlp-task/leaderboards

**Best of Luck** 🎉 🎉
