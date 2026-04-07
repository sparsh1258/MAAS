from tasks.task_1_easy import TASK_OBSERVATION as TASK1_OBS, grade as grade_task1, get_task_prompt as prompt1, TASK_ID as T1_ID, TASK_NAME as T1_NAME
from tasks.task_2_medium import TASK_OBSERVATION as TASK2_OBS, grade as grade_task2, get_task_prompt as prompt2, TASK_ID as T2_ID, TASK_NAME as T2_NAME
from tasks.task_3_hard import TASK_OBSERVATION as TASK3_OBS, grade as grade_task3, get_task_prompt as prompt3, TASK_ID as T3_ID, TASK_NAME as T3_NAME

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
]