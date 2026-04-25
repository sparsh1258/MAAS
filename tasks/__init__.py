from tasks.task_1_easy import TASK_OBSERVATION as TASK1_OBS, grade as grade_task1, get_task_prompt as prompt1, TASK_ID as T1_ID, TASK_NAME as T1_NAME
from tasks.task_2_medium import TASK_OBSERVATION as TASK2_OBS, grade as grade_task2, get_task_prompt as prompt2, TASK_ID as T2_ID, TASK_NAME as T2_NAME
from tasks.task_3_hard import TASK_OBSERVATION as TASK3_OBS, grade as grade_task3, get_task_prompt as prompt3, TASK_ID as T3_ID, TASK_NAME as T3_NAME
from tasks.task_4_gestational_diabetes import TASK_OBSERVATION as TASK4_OBS, grade as grade_task4, get_task_prompt as prompt4, TASK_ID as T4_ID, TASK_NAME as T4_NAME
from tasks.task_5_anemia import TASK_OBSERVATION as TASK5_OBS, grade as grade_task5, get_task_prompt as prompt5, TASK_ID as T5_ID, TASK_NAME as T5_NAME
from tasks.task_6_fetal_distress import TASK_OBSERVATION as TASK6_OBS, grade as grade_task6, get_task_prompt as prompt6, TASK_ID as T6_ID, TASK_NAME as T6_NAME
from tasks.task_7_preterm_risk import TASK_OBSERVATION as TASK7_OBS, grade as grade_task7, get_task_prompt as prompt7, TASK_ID as T7_ID, TASK_NAME as T7_NAME

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
]
