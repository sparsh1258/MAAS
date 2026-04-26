from tasks.task_1_easy import TASK_OBSERVATION as TASK1_OBS, grade as grade_task1, get_task_prompt as prompt1, TASK_ID as T1_ID, TASK_NAME as T1_NAME
from tasks.task_2_medium import TASK_OBSERVATION as TASK2_OBS, grade as grade_task2, get_task_prompt as prompt2, TASK_ID as T2_ID, TASK_NAME as T2_NAME
from tasks.task_3_hard import TASK_OBSERVATION as TASK3_OBS, grade as grade_task3, get_task_prompt as prompt3, TASK_ID as T3_ID, TASK_NAME as T3_NAME
from tasks.task_4_gestational_diabetes import TASK_OBSERVATION as TASK4_OBS, grade as grade_task4, get_task_prompt as prompt4, TASK_ID as T4_ID, TASK_NAME as T4_NAME
from tasks.task_5_anemia import TASK_OBSERVATION as TASK5_OBS, grade as grade_task5, get_task_prompt as prompt5, TASK_ID as T5_ID, TASK_NAME as T5_NAME
from tasks.task_6_fetal_distress import TASK_OBSERVATION as TASK6_OBS, grade as grade_task6, get_task_prompt as prompt6, TASK_ID as T6_ID, TASK_NAME as T6_NAME
from tasks.task_7_preterm_risk import TASK_OBSERVATION as TASK7_OBS, grade as grade_task7, get_task_prompt as prompt7, TASK_ID as T7_ID, TASK_NAME as T7_NAME
from tasks.task_8_preeclampsia_watch import TASK_OBSERVATION as TASK8_OBS, grade as grade_task8, get_task_prompt as prompt8, TASK_ID as T8_ID, TASK_NAME as T8_NAME
from tasks.task_9_history_noise_low_risk import TASK_OBSERVATION as TASK9_OBS, grade as grade_task9, get_task_prompt as prompt9, TASK_ID as T9_ID, TASK_NAME as T9_NAME
from tasks.task_4_multiturn_easy import TASK_ID as MT4_ID, TASK_NAME as MT4_NAME, TASK_TRAJECTORY_ID as MT4_TRAJ, grade as grade_mt4, get_task_prompt as prompt_mt4
from tasks.task_5_multiturn_hard import TASK_ID as MT5_ID, TASK_NAME as MT5_NAME, TASK_TRAJECTORY_ID as MT5_TRAJ, grade as grade_mt5, get_task_prompt as prompt_mt5

TASKS = [
    {
        "id": T1_ID,
        "name": T1_NAME,
        "difficulty": "easy",
        "observation": TASK1_OBS,
        "grade": grade_task1,
        "prompt": prompt1,
    },
    {
        "id": T2_ID,
        "name": T2_NAME,
        "difficulty": "medium",
        "observation": TASK2_OBS,
        "grade": grade_task2,
        "prompt": prompt2,
    },
    {
        "id": T3_ID,
        "name": T3_NAME,
        "difficulty": "hard",
        "observation": TASK3_OBS,
        "grade": grade_task3,
        "prompt": prompt3,
    },
    {
        "id": T4_ID,
        "name": T4_NAME,
        "difficulty": "medium",
        "observation": TASK4_OBS,
        "grade": grade_task4,
        "prompt": prompt4,
    },
    {
        "id": T5_ID,
        "name": T5_NAME,
        "difficulty": "medium",
        "observation": TASK5_OBS,
        "grade": grade_task5,
        "prompt": prompt5,
    },
    {
        "id": T6_ID,
        "name": T6_NAME,
        "difficulty": "hard",
        "observation": TASK6_OBS,
        "grade": grade_task6,
        "prompt": prompt6,
    },
    {
        "id": T7_ID,
        "name": T7_NAME,
        "difficulty": "hard",
        "observation": TASK7_OBS,
        "grade": grade_task7,
        "prompt": prompt7,
    },
    {
        "id": T8_ID,
        "name": T8_NAME,
        "difficulty": "medium",
        "observation": TASK8_OBS,
        "grade": grade_task8,
        "prompt": prompt8,
    },
    {
        "id": T9_ID,
        "name": T9_NAME,
        "difficulty": "medium",
        "observation": TASK9_OBS,
        "grade": grade_task9,
        "prompt": prompt9,
    },
]

MULTITURN_TASKS = [
    {
        "id": MT4_ID,
        "name": MT4_NAME,
        "difficulty": "easy",
        "trajectory_id": MT4_TRAJ,
        "grade": grade_mt4,
        "prompt": prompt_mt4,
        "mode": "multiturn",
    },
    {
        "id": MT5_ID,
        "name": MT5_NAME,
        "difficulty": "hard",
        "trajectory_id": MT5_TRAJ,
        "grade": grade_mt5,
        "prompt": prompt_mt5,
        "mode": "multiturn",
    },
]
