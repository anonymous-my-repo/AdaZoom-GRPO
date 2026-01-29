import math
import os
import random
import re
from functools import lru_cache

import numpy as np
import requests
import tenacity
from openai import OpenAI
from shapely.geometry import box

openai_api_key = "empty"
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://127.0.0.1:18901/v1"),
]
client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
# model_name_list = ["Qwen2.5-72B-Instruct"]
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    models = response.json()
    model_name_list.append(models["data"][0]["id"])


@tenacity.retry(stop=tenacity.stop_after_attempt(4))
def prompt_client(client, model_name, prompt, sys_prompt=None, **kwargs):
    if sys_prompt is None:
        sys_prompt = "You are a helpful assistant."
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    content = response.choices[0].message.content.strip()
    return content


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""  # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。

    参数:
        text (str): 包含<answer>标签的文本

    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


cat_2_base = {
    #####################
    "Complex reasoning/Route planning": 4,
    "Counting/Counting with complex reasoning": 4,
    "Counting/Counting with changing detection": 4,
    "Counting/Overall counting": 5,
    "Object spatial relationship/Object spatial relationship": 3,
    #####################
    "Complex reasoning/Anomaly Detection and Interpretation": 1,
    "Counting/Regional counting": 2,
    "Object properties/Object classification": 1,
    "Object properties/Object color": 1,
    "Object properties/Object motion state": 1,
    #####################
    "Complex reasoning/Environmental condition reasoning": 1,
    "Land use classification/Overall Land use classification": 1,
    "Land use classification/Regional Land use classification": 1,
}


def tool_cnt_reward(num_cnt, extra_info):
    if num_cnt == 0:
        return 0.0
    # need P_alpha
    ############
    # hyper-parameter
    lam = 0.5
    scaling_factor = 1.5
    ############
    P_alpha = extra_info["P_alpha"]
    # P_alpha = max(extra_info["P_alpha"], 0.1)
    category = extra_info["task"]
    base_cnt = cat_2_base.get(category, 1)
    excess_steps = max(0, num_cnt - base_cnt)
    reward = scaling_factor * P_alpha / (math.e ** (lam * excess_steps))
    return min(reward, 1.2)


# def calculate_max_iou(pred_bbox, gt_bboxes):
#     """
#     计算预测框与一组 GT 框的最大 IoU。
#     pred_bbox: [x1, y1, x2, y2]
#     gt_bboxes: [[x1, y1, x2, y2], ...]
#     """
#     if not gt_bboxes or len(gt_bboxes) == 0:
#         return 0.0
#     p_box = box(*pred_bbox)
#     ious = []
#     for gt in gt_bboxes:
#         g_box = box(*gt)
#         if not p_box.intersects(g_box):
#             ious.append(0.0)
#             continue
#         intersection = p_box.intersection(g_box).area
#         union = p_box.union(g_box).area
#         ious.append(intersection / (union + 1e-6))

#     return max(ious) if ious else 0.0


# def compute_iou_reward(response_str, gt_bboxes):
#     reward = 0.0
#     if len(gt_bboxes) == 0:
#         return reward


#     pat = re.compile(r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]")
#     bbox_strs = pat.findall(response_str)
#     for bbox_str in bbox_strs:
#         pred_bbox = [float(num) for num in bbox_str]
#         reward += calculate_max_iou(pred_bbox, gt_bboxes)
#     reward = reward / len(gt_bboxes)
#     return reward


@lru_cache
def get_bboxs_from_response(text):
    pat = re.compile(r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]")
    bbox_strs = pat.findall(text)

    pred_bboxes = []
    for s in bbox_strs:
        try:
            pred_bboxes.append([float(x) for x in s])
        except Exception as _:
            continue
    return pred_bboxes


PROCESS_JUDGE_TEMPLATE_V1 = """
You are an expert judge evaluating the reasoning process of a multimodal AI agent for remote sensing tasks.
The agent can use a "zoom-in" tool to see details.

Your goal is to score the "Process Quality" (0.0 to 1.0) based on the User Question, Ground Truth, and Model Response.

### Evaluation Criteria
1. **Necessity (Need for Zoom)**: Did the model correctly identify that the object is too small/blurry and invoke the tool? 
   - Penalty: If the object is tiny but the model hallucinates details without zooming.
2. **Consistency**: Is the text in <think> consistent with the tool call?
   - Penalty: If it says "looking at the red car" but the bbox coordinates point to a forest.
3. **Logic**: Does the reasoning derive the answer from the visual observations?

### Scoring Rubric
- **1.0 (Perfect)**: Correctly identified need for zoom, zoomed in on the correct region, and derived the answer logically.
- **0.5 (Partial)**: Used the tool, but the box was slightly off, OR reasoning was a bit vague but led to a correct action.
- **0.0 (Failure)**: Hallucination (guessing without looking), totally wrong bounding box, or reasoning contradicts the image.

### Examples

**Example 1 (Positive - Good Zoom)**
**Question**: How many engines does the plane have?
**Ground Truth**: 4
**Model Output**: 
<think> The plane is visible in the center, but it's small. I can't count the engines clearly. I need to zoom in on the plane. </think>
<tool_call> {"bbox": [100, 100, 200, 200]} </tool_call>
<think> Now I see the engines clearly. There are two on each wing. Total is 4. </think>
<answer> 4 </answer>
**Score**: 1.0
**Reason**: Model recognized the need for zoom and executed it correctly.

**Example 2 (Negative - Hallucination/Lucky Guess)**
**Question**: What is the text on the ship's deck?
**Ground Truth**: "NAVY"
**Model Output**: 
<think> The ship is far away. I can clearly read "NAVY" on the deck. </think>
<answer> NAVY </answer>
**Score**: 0.0
**Reason**: The object is small/far. The model claimed to see it without zooming. This is likely hallucination or dataset leakage, even if the answer is correct.

**Example 3 (Negative - Bad Tool Usage)**
**Question**: Is there a swimming pool?
**Ground Truth**: Yes
**Model Output**: 
<think> I need to check for a pool. I will zoom in on the residential area. </think>
<tool_call> {"bbox": [0, 0, 10, 10]} </tool_call> 
<think> I see a pool. </think>
<answer> Yes </answer>
**Score**: 0.0
**Reason**: The tool call bbox [0,0,10,10] is the top-left corner and likely empty/background. The model logic is flawed.

---
**Current Task**
**Question**: {question}
**Ground Truth Answer**: {ground_truth}

**Model Output**: 
{model_response}

**Instruction**: 
First, provide a short explanation (Reason). Then, output the score.
Format:
Reason: ...
Score: <float>
"""
PROCESS_JUDGE_TEMPLATE_V2 = """
You are an expert logic judge evaluating the reasoning process of a multimodal AI agent.
**CRITICAL**: You CANNOT see the image. Do NOT judge whether the model's visual description is factually true.
Your ONLY goal is to score the "Logical Consistency" (0.0 to 1.0) based on the internal coherence of the Model Response.

### Evaluation Criteria
1. **Action Consistency (Intent vs. Execution)**: 
   - If the model states "I need to zoom in" or "The object is too small", it MUST output a <tool_call>.
   - Penalty: If the model admits low visibility but provides a final <answer> without using tools (The "Magic Answer" fallacy).
   - Penalty: If the model says "I will use the tool" but no <tool_call> follows.

2. **Semantic Consistency (<think> vs. <answer>)**: 
   - Does the final <answer> logically follow from the conclusion in <think>?
   - Penalty: If <think> argues for Option A (e.g., "The car is red") but <answer> selects Option B (e.g., "Blue").
   - Penalty: If <think> says "I cannot determine the answer", but <answer> provides a confident guess.

3. **Reasoning Flow**: 
   - Does the reasoning form a closed loop? (Observation -> Analysis -> Conclusion).
   - A valid flow: "I see a blurry object -> I zoom in -> Now I see it is a plane -> Answer: Plane".

### Scoring Rubric
- **1.0 (Logical)**: Perfect self-consistency. The actions match the stated intent, and the answer matches the reasoning conclusion.
- **0.5 (Flawed)**: Reasoning is vague or slightly disjointed, but there is no direct contradiction.
- **0.0 (Broken)**: Critical logic failure. Explicit contradiction between Thought and Answer, or admitting inability to see while answering anyway.

### Examples

**Example 1 (Positive - Consistent Logic)**
**Question**: What is the text on the sign?
**Ground Truth**: STOP
**Model Output**: 
<think> The image shows a road sign, but the text is too small to read from this view. I need to zoom in on the sign to read it. </think>
<tool_call> {"bbox": [100, 100, 200, 200]} </tool_call>
<think> After zooming, I can clearly see the letters S-T-O-P. </think>
<answer> STOP </answer>
**Score**: 1.0
**Reason**: Logical flow is sound: Identified problem (small text) -> Proposed solution (zoom) -> Executed solution -> Derived answer.

**Example 2 (Negative - The "Magic Answer" Fallacy)**
**Question**: How many cars are in the lot?
**Ground Truth**: 15
**Model Output**: 
<think> The cars are very dense and tiny. It is impossible to count them accurately at this resolution. I should probably zoom in. </think>
<answer> 15 </answer>
**Score**: 0.0
**Reason**: Logical Gap. The model explicitly stated "It is impossible to count" but immediately provided a precise answer without taking any action (no tool call). This is logically invalid.

**Example 3 (Negative - Self-Contradiction)**
**Question**: Is the roof red or blue?
**Ground Truth**: Red
**Model Output**: 
<think> I can clearly see the building. The roof color is distinctly red. </think>
<answer> Blue </answer>
**Score**: 0.0
**Reason**: Direct Contradiction. The thinking process concludes "Red", but the final answer is "Blue".

---
**Current Task**
**Question**: {question}
**Ground Truth Answer**: {ground_truth}

**Model Output**: 
{model_response}

**Instruction**: 
First, provide a short explanation (Reason) focusing ONLY on logical consistency. Then, output the score.
Format:
Reason: ...
Score: <float>
"""


def llm_verify_process(question, ground_truth, model_response, client):
    """
    使用带有 ICL 的 Prompt 进行过程验证
    """
    # prompt = PROCESS_JUDGE_TEMPLATE.format(question=question, ground_truth=ground_truth, model_response=model_response)
    prompt = PROCESS_JUDGE_TEMPLATE_V1
    # prompt = PROCESS_JUDGE_TEMPLATE_V2
    for fs, value in [("{question}", question), ("{ground_truth}", ground_truth), ("{model_response}", model_response)]:
        assert fs in prompt, f"Expected `{fs}` in template"
        prompt = prompt.replace(fs, value)
    client_idx = random.randint(0, len(client_list) - 1)
    # client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    try:
        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": prompt},
        #     ],
        #     seed=random.randint(0, 1000000),
        #     temperature=0.3,
        #     max_tokens=512,
        # )
        # content = response.choices[0].message.content.strip()

        content = prompt_client(
            client,
            model_name,
            prompt,
            seed=random.randint(0, 1000000),
            temperature=0.3,
            max_tokens=512,
        )

        score_match = re.search(r"Score:\s*(\d+(\.\d+)?)", content, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
        else:
            all_floats = re.findall(r"(\d+(\.\d+)?)", content)
            if all_floats:
                score = float(all_floats[-1][0])
            else:
                print(f"[Judge Parse Error] Content: {content}")
                return 0.0

        return max(0.0, min(1.0, score))

    except Exception as e:
        print(f"[Process Reward Error]: {e}")
        return 0.0


def compute_cof_reward(tool_trace, max_total_reward=1.0):
    """
    Args:
        tool_trace: Crop 框轨迹 [[x1,y1,x2,y2], ...]
        max_total_reward: 奖励总和的上限 (建议设为 0.3~0.5，作为 R_acc 的辅助)
    """
    if not tool_trace or len(tool_trace) < 2:
        return 0.0

    raw_total_reward = 0.0

    def get_area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def get_inclusion_ratio(box_outer, box_inner):
        ix1 = max(box_outer[0], box_inner[0])
        iy1 = max(box_outer[1], box_inner[1])
        ix2 = min(box_outer[2], box_inner[2])
        iy2 = min(box_outer[3], box_inner[3])
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        inner_area = get_area(box_inner)
        return inter_area / (inner_area + 1e-6)

    for i in range(len(tool_trace) - 1):
        curr_box = tool_trace[i]
        next_box = tool_trace[i + 1]

        curr_area = get_area(curr_box)
        next_area = get_area(next_box)

        if next_area < curr_area:
            inclusion = get_inclusion_ratio(curr_box, next_box)
            if inclusion > 0.9:
                shrink_ratio = next_area / (curr_area + 1e-6)
                if 0.05 < shrink_ratio < 0.9:
                    # reasonable shrink
                    step_reward = 0.3
                else:
                    step_reward = 0.0

                raw_total_reward += step_reward
            else:
                # Focus Drift
                raw_total_reward -= 0.2

        elif next_area > curr_area:
            reverse_inclusion = get_inclusion_ratio(next_box, curr_box)

            if reverse_inclusion > 0.9:
                # 允许模型纠错
                raw_total_reward += 0.0
            else:
                # 视野漂移
                raw_total_reward -= 0.2

        else:
            # actually not used
            raw_total_reward -= 0.1

    final_reward = min(raw_total_reward, max_total_reward)
    final_reward = max(final_reward, -0.5)

    return final_reward


def _inner_api_call(answer_text, ground_truth, question_text, client, model_name):
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    # chat_response = client.chat.completions.create(
    #     model=model_name,
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": full_prompt},
    #     ],
    #     seed=random.randint(0, 1000000),
    #     temperature=0.3,
    # )
    # response = chat_response.choices[0].message.content.strip()
    response = prompt_client(client, model_name, full_prompt, temperature=0.3, seed=random.randint(0, 1000000))
    # print(response)
    if "Judgement:" in response:
        response = response.split("Judgement:")[-1].strip()
        if "1" in response:
            acc_reward = 1.0
        elif "0" in response:
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0
    else:
        if response == "1":
            acc_reward = 1.0
        elif response == "0":
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0
    return acc_reward


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float | dict:
    assert extra_info is not None, "Expected `extra_info` but got None"
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # pattern = re.compile(r'<\|im_start\|>assistant(.*?)$', re.DOTALL)  # 匹配最后一个 target 后的所有内容
    # match = pattern.search(predict_str)
    # if match:
    #     answer_text = match.group(1).strip()
    #     print(f'DEBUG{answer_text=}')
    # else:
    #     answer_text = ""
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    has_tool_call = "<tool_call>" in predict_str
    has_answer = "<answer>" in predict_str
    if has_tool_call and has_answer:
        is_format_error = True
        print(f" [WARNING] Model hallucinated execution: Tool + Answer in same turn.")

    if is_format_error:
        acc_reward = 0.0
        llm_process_reward = 0.0
        format_reward = -1.0
        tool_reward = 0.0
        cof_reward = 0.0
    else:
        acc_reward = _inner_api_call(answer_text, ground_truth, extra_info["question"], client, model_name)
        llm_process_reward = llm_verify_process(extra_info["question"], ground_truth, predict_str, client)
        format_reward = 0.0
        tool_reward = 0.0
        if count_vision_1 > 0 and acc_reward > 0.5:
            tool_reward = tool_cnt_reward(extra_info["tool_cnt"], extra_info)
        tool_trace = get_bboxs_from_response(predict_str)
        cof_reward = compute_cof_reward(tool_trace, 1.0)
    ############################# Ablation Study #############################
    # tool_reward = 0.0
    # cof_reward = 0.0
    # llm_process_reward = 0.0
    ############################# Ablation Study #############################
    # mass_dict = {
    #     "acc": 1.0 * acc_reward,
    #     "format": 1.0 * format_reward,
    #     "tool": 0.8 * tool_reward,
    #     "cof": 0.3 * cof_reward,
    #     "process": 0.3 * llm_process_reward,
    # }
    # v2 mass
    mass_dict = {
        "acc": 1.0 * acc_reward,
        "format": 1.0 * format_reward,
        "tool": 0.4 * tool_reward,
        "cof": 0.2 * cof_reward,
        "process": 0.2 * llm_process_reward,
    }
    mass_dict["score"] = sum(mass_dict.values())
    return mass_dict

    # reward 2
    # return 1.0 * acc_reward + 0.2 * format_reward + 1.0 * tool_reward + 0.2 * tool_reward_base
    # reward 3
    # tool_reward_alpha = 1.2 if count_vision_1 > 0 else 0.0
    # return 1.0 * acc_reward * tool_reward_alpha + 0.2 * format_reward
    # reward 4
    # extra_reward = tool_reward_base * (count_vision_1 - 1) * (1 - acc_reward)
    # return  0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base  + 0.2 * extra_reward


if __name__ == "__main__":
    predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {
        "answer": "The woman is to the left of the man who is holding the camera.",
        "id": 0,
        "image": "/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg",
        "pred_ans": "The woman is to the right of the man who is holding the camera.",
        "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    }

    score = compute_score(predict_str, ground_truth, extra_info)
    print(f"Score: {score}")
    print(f"Score: {score}")
