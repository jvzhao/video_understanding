Top Authors

Chang D. Yoo : KAIST AIM Laboratory

Haruo Takemura, Nakashima, Yuta: Osaka University

Yuan-Fang Li : Monash University

Lianli Gao , Song Jingkuan, HengTao Shen : UESTC

Zhou Zhao, Jun XIAO, Yueting Zhuang, Deng Cai, Fei Wu, Siliang Tang: ZJU

Truyen Tran : Deakin University

[BERT Representations for Video Question Answering](https://www.semanticscholar.org/paper/39b2d8b8233a53dc7eadb819c52213369dff8648)(将BERT引入到Video QA任务中)

数据集:

The MovieQA dataset for the video QA mode consists of 140 movies with 6,462 QA pairs . Each question is coupled with a set of five possible answers; one correct and four incorrect answers. A QA model should choose a correct answer for a given question only using provided video clips and subtitles. The average length of a video clip is 202 seconds. If there are multiple video clips given in one question, we link them together into a single video clip. The number of QA pairs in train/val/test are 4318/886/1258, respectively.

The PororoQA dataset has 171 episodes with 8,834 QA pairs . Like MovieQA, each question has one correct answer sentence and four incorrect answer sentences. One episode consists of a video clip of 431 seconds average length. For experiments, we split all 171 episodes into train(103 ep.)/val(34 ep.)/test(34 ep.) sets. The number of QA pairs in train/val/test are 5521/1955/1437, respectively. Unlike the MovieQA, the PororoQA has supporting fact labels that indicate which of the frames and captions of the video clip contain correct answer information, and description set. However, because our model does not use any supporting fact label or description, we do not use them in the experiment.

TVQA Dataset TVQA dataset (Lei et al., 2018) consists of video frames, subtitles, and questionanswer pairs from 6 TV shows. The number of examples for train/validation/test-public dataset are 122,039/15,253/7,623. Each example has five candidate answers with one of them the ground-truth. So, TVQA is a classification task, in which models select one from the five candidate answers, and models can be evaluated on the accuracy metric.

SVQA: This dataset is a benchmark for multi-step reasoning. Resembling the CLEVR dataset [15] for traditional visual question answering task, SVQA provides long questions with logical structures along with spatial and temporal interactions between objects. SVQA was designed to mitigate several drawbacks of current Video QA datasets including language bias and the shortage of compositional logical structure in questions. It contains 120K QA pairs generated from 12K videos that cover a number of question types such as count, exist, object attributes comparison and query.

TGIF-QA: This is currently the largest dataset for Video QA, containing 165K QA pairs collected from 72K animated GIFs. This dataset covers four sub-tasks mostly to address the unique properties of video including repetition count, repeating action, state transition and frame QA. Of the four tasks, the first three require strong spatio-temporal reasoning abilities. Repetition Count: This is one of the most challenging tasks in Video QA where machines are asked count the repetitions of an action. For example, one has to answer questions like “How many times does the woman shake hips?”. This is defined as an open-ended task with 11 possible answers in total ranging from 0 to 10+. Repeating Action: This is a multiple choice task of five answer candidates corresponding to one question. The task is to identify the action that is repeated for a given number of times in the video (e.g. “what does the dog do 4 times?”). State Transition: This is also a multiplechoice task asking machines to perceive the transition between two states/events.  There are certain states characterized in the dataset including facial expressions, actions, places and object properties. Questions like “What does the woman do before turn to the right side?” and “What does the woman do after look left side?” aim at identifying previous state and next state, respectively. Frame QA: This task is akin to the traditional visual QA where the answer to a question can be found in one of the frames in a video. None of temporal relations is necessary to answer questions

MSVD-QA [38] dataset contains 1, 970 short clips and 50, 505 question answer pairs. The questions are composed of five types, including what, who, how, when, and where.

MSRVTT-QA [39] dataset contains 10K videos and 243K question answer pairs. While types of questions are the same with MSVD-QA dataset, the contents of the videos in MSRVTT-QA are more complex and the lengths of the videos are much longer from 10 to 30 seconds



