
import argparse
import copy
import json
import logging
import multiprocessing
import os
from glob import glob
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import ray
import transformers
import time

from helper import *
from model_utils import LM

default_sys = "You are a helpful assistant."
boxed_sysprompt = "Please reason step by step, and put your final answer within \\boxed{}."

templates = {
# serves as the standard inference without reference
"standard_inference_en": """You are an expert in many fields. Suppose you will give a specific final response, I need you to also write down the thought process behind this solution.
Here is a task:
{}

Now, you need to think aloud and brainstorm in the mind. The thinking process involves thoroughly exploring questions through a systematic long thinking process. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Present your complete thought process within a single and unique `<think></think>` tag.

Your thought process must adhere to the following requirements:

1.  **Narrate in the first-person as if you are thinking aloud and brainstorming**
    Stick to the narrative of "I". Imagine you are brainstorming and thinking in the mind. Use verbalized, simple language.

2.  **Unify the thinking process and the writing solution:**
    Your thought process must precisely correspond to a part of the writing solution. The reader should be able to clearly see how your thoughts progressively "grew" into the finished piece, making the copy feel like the inevitable product of your thinking.

3.  **Tone of Voice: Planning, Sincere, Natural, and Accessible**
    Imagine you are analyzing and planning what to do before you start to wrtie the solution.  Your language should be plain and easy to understand, avoiding obscure professional jargon to explain complex thought processes clearly.

4.  **Logical Flow: Clear and Progressive**
    
5.  **Thinking Framework for deep thinking**
    To ensure your thinking is clear and deep, to showcase your thinking and planning to fulfill the task, below is what you might cover when you are thinking aloud and brainstorming.

    Understanding the user intent and the task: Before putting pen to paper, I need to thoroughly consider the fundamental purpose of the writing. I first need to discern the user's true goal behind their literal request. Next, I will consider: Who am I talking to? I will create a precise profile of the target reader, understanding their pain points, aspirations, and reading context. Then, I will establish the Core Objective: What specific emotional, cognitive, and behavioral changes do I most want the reader to experience after reading? 
    
    Establishing the content: I need to brainstorm a core creative idea and communication strategy centered around my objective. Then, I will think about what content and key information I need to convey to the reader to fulfill the writing task, and what source materials this will involve.

    Building the structure: I need to design a clear narrative path for the reader, like a "blueprint." First, I will plan the article's skeleton (e.g., using a framework like the Golden Circle "Why-How-What," the AIDA model "Attention-Interest-Desire-Action," or a narrative structure "Beginning-Development-Climax-Resolution"). Then, I will plan the key modules: How will the introduction hook the reader? How will the body be layered and the arguments arranged? How will the conclusion summarize, elevate the message, and provide a clear Call to Action (CTA)?

    Draft: unless it is a really easy request, otherwise I need to consider writing a draft based on the plan above, before you give the final writing solution.  I will translate my plan into paragraphs, considering the key points, content, and sentence structure for each. This initial draft should look like a prototype of the target text. This draft will be way shorter than the final writing solution within controlled length, but it must also avoid being too vague or general or simply copying the final text. I will begin writing this draft after a `--- The Draft ---` separator at the end of my thinking process. The draft will be included in the same `<think></think>` block. After writing the draft, I will further critique what can be improved, and analyze what details can be enriched (and hence make it more likely to eventually arrive at the given solution)
    
6. Throughout the thinking process, I want to involve deep thinking and planning, and use deliberate self-critique/self-reflection in my thinking process. Trigger these by frequently using patterns such as `wait`, `maybe`, `let me`, etc. For example:
    - Hmm, maybe .. (other concrete thinking regarding the given request)
    - Let me think .. 
    - Wait no ..
    - But wait ..(might find something wrong with your previous thoughts)
    - Wait, that's a bit ..(reflections about previous decisions). Let me think .. (are thinking of other possibilities)
    - Wait, the user said ..(backtracking of previous information). So ..
    - Hmm...Alternatively, maybe ..(branching on other possibilities)
    - But ..

Now record your clear, complete, and logical thinking process within `<think></think>` tags. 
In the thinking process, make sure NO PAST TENSES, NO PAST TENSES, because this is the thought process before you are to write a final solution. You are planning what you will and you need to do.
Imagine you're thinking aloud and brainstorming. Write it as an internal monologue or a stream of consciousness. Do not use bullet points, numbers, or formal section headings. 
""",
"standard_inference_cn": """你是一名各种领域专家，设想有一个用户请求和一个回答。现在你需要针对回答解释思考过程，特别是如何针对这个请求进行深入思考、深入规划的。
下面是用户请求：\n{}

现在你需要头脑风暴，在（单独且唯一的）`<think></think>`标签中呈现你的完整思考过程。

思考过程必须遵循以下要求：
关于叙述视角：使用第一人称，想象你在大脑里头脑风暴，演绎自己的创作思考过程。用口语化的表述和通俗的语言。

关于语言风格：未来时、真诚、自然、易懂
设想你在动笔前分析和规划的思考过程，所以应该是用未来、计划性或者“我应该”这种语气。请用真诚、坦率的口吻，像一位经验丰富的前辈在传授经验。语言要平实、易懂，避免使用晦涩的专业术语，把复杂的思考过程说明白。

关于思考的逻辑：清晰、层层递进
整个思考过程需要展现出清晰的因果链条，层层递进，解释“为什么这么想”以及“这样做预计会带来什么效果”。思考过程中，基于上面的写作框架中的核心步骤，不断进行细节拆分，使用多样化的逻辑连接词，例如“首先”、“其次”、“然后”、来逐步递进思考过程，完善细节。避免反复使用相同的连接词。

思维框架：
为了让思考过程清晰、有深度，我建议你采用下面的创作框架来组织思路。这能让你的思考过程更接近一位真实专家的工作流：

首先思考，我为何而写？在动笔前，我会先彻底想清楚写作的根本目的。我需要先洞察用户字面需求背后的真实目标，接着思考：我在对谁说话？精准描绘出目标读者的画像，理解他们的痛点、渴望和阅读场景。然后，确立核心目标： 我最希望读者读完后，在情感、认知和行动上发生什么具体变化？

然后确立内容，我要写什么？我需要围绕写作目标构思核心创意和沟通策略，规划内容。然后思考，为了完成用户请求，我需要向读者传递包括哪些内容和关键信息，分别涉及到什么素材。

接着搭建结构，思考我要怎么写？我需要设计一个清晰的行文路径，像“施工图”一样引导读者。首先，我需要规划文章骨架（例如：黄金圈法则 "Why-How-What"、AIDA 模型 "Attention-Interest-Desire-Action"、故事结构 "开端-发展-高潮-结尾"等）。然后，我要考虑布局关键模块： 开头如何破题？主体如何分层展开、安排论据？结尾如何总结、升华，并给出清晰的行动号召 (Call to Action)？

再然后，除非是很显然很容易的请求，否则考虑先写一个草稿。我需要落实到每一个段落，具体考虑有什么要点，写什么内容，句子如何组织。要让这份初稿看起来像是上面的文案的雏形，但是要避免照搬上面的文案，又要避免语言笼统。`--- 草稿 ---`分割线后开始写你的草稿，但是草稿部分和上面的思考过程都要放在同一个`<think></think>`标签内
草稿结束后再次思考有什么可以进一步调整的细节、或者进一步优化的地方，这也是为什么要和真正最终回答有所区分，应当是最终回答的雏形

为了充分思考和深入推理，我会多使用自我反思和自我评判来进一步展开细节、分支其他方面或者回溯思考之前的一些陈述。我会频繁利用一些触发自我反思和自我批判的词语：“不过”、“或者”、“可能”，用这些词来触发更加细节、更加深入的思考，下面是一些例子：
    - 嗯，也许……（关于给定请求的其他具体思考）
    - 让我想想……
    - 等等，不对……
    - 不过等等……（可能会发现你之前的想法有问题）
    - 等等，这有点……（对先前决定的反思）。让我想想……（正在思考其他可能性）
    - 等等，用户说……（回溯之前的信息）。所以……
    - 嗯……或者，也许……（思考其他分支可能性）
    - 但是……
格式上，将清晰完整有逻辑的思考过程在`<think></think>`标签中记录。
在思考过程中，确保不要使用过去时，不要使用过去时，因为这是在你写最终解决方案之前的思考过程。你正在计划你将要做什么和需要做什么。
想象你正在出声思考和进行头脑风暴。把它写成内心独白或意识流。不要使用项目符号、编号或正式的章节标题。下面，设想你是首次拿到这个用户请求，然后开始你的思考（不要暗示你在解释一个回答。
""",
"initial_thinking_en": """You are an expert in many fields. Suppose you will give a specific final response, I need you to also write down the thought process behind this solution.
Here is a task:
{}

Here is the solution you will create:
{}

Now, you need to write down the thinking process behind this solution, as if you are thinking aloud and brainstorming in the mind. The thinking process involves thoroughly exploring questions through a systematic long thinking process. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Present your complete thought process within a single and unique `<think></think>` tag.

Your thought process must adhere to the following requirements:

1.  **Narrate in the first-person as if you are thinking aloud and brainstorming**
    Stick to the narrative of "I". Imagine you are brainstorming and thinking in the mind. Use verbalized, simple language.

2.  **Unify the thinking process and the writing solution:**
    Your thought process must precisely correspond to a part of the writing solution. The reader should be able to clearly see how your thoughts progressively "grew" into the finished piece, making the copy feel like the inevitable product of your thinking.

3.  **Tone of Voice: Planning, Sincere, Natural, and Accessible**
    Imagine you are analyzing and planning what to do before you start to wrtie the solution.  Your language should be plain and easy to understand, avoiding obscure professional jargon to explain complex thought processes clearly.

4.  **Logical Flow: Clear and Progressive**
    
5.  **Thinking Framework for deep thinking**
    To ensure your thinking is clear and deep, to showcase your thinking and planning to fulfill the task, below is what you might cover when you are thinking aloud and brainstorming.

    Understanding the user intent and the task: Before putting pen to paper, I need to thoroughly consider the fundamental purpose of the writing. I first need to discern the user's true goal behind their literal request. Next, I will consider: Who am I talking to? I will create a precise profile of the target reader, understanding their pain points, aspirations, and reading context. Then, I will establish the Core Objective: What specific emotional, cognitive, and behavioral changes do I most want the reader to experience after reading? 
    
    Establishing the content: I need to brainstorm a core creative idea and communication strategy centered around my objective. Then, I will think about what content and key information I need to convey to the reader to fulfill the writing task, and what source materials this will involve.

    Building the structure: I need to design a clear narrative path for the reader, like a "blueprint." First, I will plan the article's skeleton (e.g., using a framework like the Golden Circle "Why-How-What," the AIDA model "Attention-Interest-Desire-Action," or a narrative structure "Beginning-Development-Climax-Resolution"). Then, I will plan the key modules: How will the introduction hook the reader? How will the body be layered and the arguments arranged? How will the conclusion summarize, elevate the message, and provide a clear Call to Action (CTA)?

    Outline: If the task output might be relatively long, I will consider writing an outline (or a draft) which naturally derives from the plan above. Specifically, the outline will ground my plan into paragraphs, summarizing the key content for each paragraph and what are the key points here, sentence structure or anything important for the paragraph. 
    I PROMISE I will NOT copy the solution I will NOT copy the solution, this outline (or draft) should only look like a prototype or outline of the target text. After finishing this outline, I will check again if there are any details or notes I should pay attention to when writing the final solution.
    I will begin writing this draft after a `--- Outline (or Draft) ---` separator at the end of my thinking process. The draft will be included in the same `<think></think>` block.
    
    
6. Throughout the thinking process, I want to involve deep thinking and planning, and use deliberate self-critique/self-reflection in my thinking process. Trigger these by regularly using patterns such as `wait`, `maybe`, `let me`, etc. For example:
    - Hmm, maybe .. (other concrete thinking regarding the given request)
    - Let me think .. 
    - Wait no ..
    - But wait ..(might find something wrong with your previous thoughts)
    - Wait, that's a bit ..(reflections about previous decisions). Let me think .. (are thinking of other possibilities)
    - Wait, the user said ..(backtracing of previous information). So ..
    - Hmm...Alternatively, maybe ..(branching on other possibilities)
    - But ..
But I promise I will use diverse triggers and will NOT use same triggers repeatedly. I will use these when analyzing user needs, establishing content and structure and when I consider alternatives, backtracing and the details. I will NOT use them when I write the draft or I am approaching the end of thinking. 

In the thinking process, make sure NO PAST TENSES, NO PAST TENSES, because this is the thought process before you are to write a final solution. You are planning what you will and you need to do.
Imagine you're thinking aloud and brainstorming. Write it as an internal monologue or a stream of consciousness. Do not use bullet points, numbers, or formal section headings. 
Now record your thinking process within `<think></think>` tags. 
""",
"initial_thinking_cn": """你是一名各种领域专家，设想有一个用户请求，你为此正在头脑风暴并且，并且把你的深入思考记录下来。
下面是用户请求：\n{}

假设下面是你会完成的文案：\n{}

现在你需要写出对应的思考过程，就像在大脑里头脑风暴。在（单独且唯一的）`<think></think>`标签中呈现你的完整思考过程。

思考过程必须遵循以下要求：
1. 关于叙述视角：使用第一人称，想象你在大脑里头脑风暴，演绎自己的创作思考过程。用口语化的表述和通俗的语言。

2. 关于思维与作品的统一：思考即作品，作品即思考
你的每一个思考步骤，都必须在最终的文案中找到精准的对应。要让读者清晰地看到，你的思考是如何一步步“长”成这篇作品的，整个复盘过程要让人觉得，这篇文案正是这些思考的必然产物。

3. 关于语言风格：未来时、真诚、自然、易懂
设想你在动笔前分析和规划的思考过程，所以应该是用未来、计划性或者“我应该”这种语气。请用真诚、坦率的口吻，像一位经验丰富的前辈在传授经验。语言要平实、易懂，避免使用晦涩的专业术语，把复杂的思考过程说明白。

4. 关于思考的逻辑：清晰、层层递进
整个思考过程需要展现出清晰的因果链条，层层递进，解释“为什么这么想”以及“这样做预计会带来什么效果”。思考过程中，基于上面的写作框架中的核心步骤，不断进行细节拆分，使用多样化的逻辑连接词，例如“首先”、“其次”、“然后”、来逐步递进思考过程，完善细节。避免反复使用相同的连接词。

5. 思维框架：
对于给定的用户请求，一个清晰、有深度、细节丰富的思考过程可能包含下面这些内容和思考方向：

为何而写？在动笔前，我会先彻底想清楚写作的根本目的。我需要先洞察用户字面需求背后的真实目标，接着思考：我在对谁说话？精准描绘出目标读者的画像，理解他们的痛点、渴望和阅读场景。然后，确立核心目标： 我最希望读者读完后，在情感、认知和行动上发生什么具体变化？

确立内容，我要写什么？我需要围绕写作目标构思核心创意和沟通策略，规划内容。然后思考，为了完成用户请求，我需要向读者传递包括哪些内容和关键信息，分别涉及到什么素材。

搭建结构，思考我要怎么写？我需要设计一个清晰的行文路径，像“施工图”一样引导读者。首先，我需要规划文章骨架（例如：黄金圈法则 "Why-How-What"、AIDA 模型 "Attention-Interest-Desire-Action"、故事结构 "开端-发展-高潮-结尾"等）。然后，我要考虑布局关键模块： 开头如何破题？主体如何分层展开、安排论据？结尾如何总结、升华，并给出清晰的行动号召 (Call to Action)？

如果是需要输出相对比较长的回答，我会考虑先写一个提纲（或者草稿），会对于参考回答进行提纲挈领，并且列出来每个段落或者部分有什么要点，写什么内容，句子如何组织。
我**绝对不会照抄绝对不会照抄**参考回答。我会让这个看起来像是一个雏形或者大纲，而不是照搬上面的文案。写完这个提纲之后，我可能会总结一下最终的回答还有没有什么细节需要主要
我会在`--- 提纲（或者草稿） ---`分割线后开始，和上面的思考过程都要放在同一个`<think></think>`标签内


6. 为了充分思考和深入推理，我会多使用自我反思和自我评判来进一步展开细节、分支其他方面或者回溯思考之前的一些陈述。我会利用一些触发自我反思和自我批判的词语：“不过”、“或者”、“可能”，用这些词来触发更加细节、更加深入的思考，下面是一些例子：
    - 嗯，也许……（关于给定请求的其他具体思考）
    - 让我想想……
    - 等等，不对……
    - 不过等等……（可能会发现你之前的想法有问题）
    - 等等，这有点……（对先前决定的反思）。让我想想……（正在思考其他可能性）
    - 等等，用户说……（回溯之前的信息）。所以……
    - 嗯……或者，也许……（思考其他分支可能性）
    - 但是……
但是我保证会多样化而且不会反复使用相同的触发词。我会在思考用户需求、确定内容和搭建结构的时候考虑使用这些，特别是需要回溯思考、分支或者展开细节的时候使用。如果是在写草稿，或者是快要完成思考的阶段，我不去使用这些触发词。

现在，想象我正在内心独白进行头脑风暴……在`<think></think>`标签中。
""",
# writing的核心是align
"segment_edits_en": """
Your task is to receive a writing assignment, a target sample text, and an existing thinking process, and then to refine and enrich a specific paragraph within that thinking process.

---> **Writing Task** \n{} 
---> **Target Text** \n{} 
---> **Thinking Process** 
{}
<replace> 
{}
</replace>
{}

Follow this three-step method to construct your response:

**Step 1:** Locate the paragraph you need to revise within the existing thinking process. In relation to the surrounding context, what is the primary function of this paragraph?

**Step 2:** Read the "Target Text" and the "paragraph to be revised" side-by-side. Ask yourself a key question: Does the thinking process reflected in this paragraph lack crucial steps, or is there content that could be further optimized and detailed to better align with the Target Text?

**Step 3:** Improve and optimize the paragraph (that represents a part of the thinking process). 
- Based on the analysis, refine the initial target paragraph into a new one, base remain the tone. Put the refinement into <refine></refine> tags.
- To help involve deep thinking and planning, consider deliberate self-critique/self-reflection in your thinking process. Trigger these by frequently using patterns such as `wait`, `maybe`, `let me`, etc. For example:
    - Hmm, maybe .. (other concrete thinking regarding the given request)
    - Let me think .. 
    - Wait no ..
    - But wait ..(might find something wrong with your previous thoughts)
    - Wait, that's a bit ..(reflections about previous decisions). Let me think .. (are thinking of other possibilities)
    - Wait, the user said ..(backtracking of previous information). So ..
    - Hmm...Alternatively, maybe ..(branching on other possibilities)
    - But ..
- If the function of the paragraph being improved is to serve as a first draft of the text, you must focus on enhancing the text's logic and completeness. The draft should not be a general outline but should express specific content and state a clear point of view. Consider whether the current draft is an appropriate prototype for the Target Text: it should be neither too vague nor a direct copy, but should reflect a foundational version.

Based on the guide above, you are to refine **only** the section marked for replacement below.
<replace>
{}
</replace>

In your response, first, present your analysis following the three-step method within `<analyze></analyze>` tags. Finally, place the corresponding, refined paragraph of the **thinking process** within `<refine></refine>` tags. 
Notes: a. Avoid repeating. Reduce the use of the same connection words, avoid repeating the same meanings over and over again. Ensure that your revised content does not repeat information from the context.
b. please keep the first a few words of the original paragraph, especially the connection words 
c. use self-critique trigger words, such as `wait`, `maybe`, `let me`, etc. 
""",
# 不能让他直接refine，还是要进行分析
"segment_edits_cn": """你会接收一个用户请求、一篇目标范文和一个已有的思考过程，然后对该思考过程的某个段落进行优化和丰富。
---> 用户请求
{}
---> 目标文案
{}
---> 思考过程
{}
----
<replace>
{}
</replace>
----
{}

遵循以下三步法来构建我的回答：

第一步：定位我需要修改的段落在现有思考过程中的位置，相对于目前的上下文而言，这个段落主要是什么功能。
第二步：并排阅读“目标文案”和“需要修改的段落”。问自己一个核心问题：需要修改的段落所反映的思考过程，是否缺少了关键的思考步骤，或者有没有可以进一步优化、进一步细化的内容，能够更好地对应到目标文案？
第三步：改进和优化现有的思考过程。
- 根据第二步中的分析，修改这个段落，注意放在<refine></refine>标签重。
- 为了充分思考和深入推理，我会多使用自我反思和自我评判来进一步展开细节、分支其他方面或者回溯思考之前的一些陈述。我会频繁利用一些触发自我反思和自我批判的词语：“不过”、“或者”、“可能”，用这些词来触发更加细节、更加深入的思考，下面是一些例子：
    - 嗯，也许……（关于给定请求的其他具体思考）
    - 让我想想……
    - 等等，不对……
    - 不过等等……（可能会发现你之前的想法有问题）
    - 等等，这有点……（对先前决定的反思）。让我想想……（正在思考其他可能性）
    - 等等，用户说……（回溯之前的信息）。所以……
    - 嗯……或者，也许……（思考其他分支可能性）
    - 但是……
- 如果需要改进的段落的功能是作为文案的初稿，务必注意改进和优化文案的逻辑性、完整性：初稿不应该是笼统的大纲，而应该具体地表达内容、陈述观点。考虑当前的初稿是否是目标文案的一个恰当的草稿：既不能太笼统，也不能照抄，而应该反映出是一个雏形。

基于上述指南，仅仅针对下面需要替换的部分进行优化。
<replace>
{}
</replace>

在下面的回答中，首先遵循三步法进行分析，放在`<analyze></analyze>`标签中，最后将我修改后的**思考过程的对应段落**放在`<refine></refine>`标签。
注意：1. 尽可能避免重复，减少反复使用的衔接词，修改后的内容不要和上下文内容有重复。
2. 务必保留段落最开始的几个词，特别是连接词或语气词。
3. 多使用反思触发词激发更深入的思考
""",
}

think_prefix = "<think>\n"

import re

def contains_chinese(text: str) -> bool:
  """
  Checks if a string contains any Chinese characters.

  Args:
    text: The input string.

  Returns:
    True if the string contains at least one Chinese character, False otherwise.
  """
  # The \u4e00-\u9fff range covers the CJK Unified Ideographs.
  # This is the most common range for Chinese characters.
  return bool(re.search(r'[\u4e00-\u9fff]', text))

@ray.remote
def generate(inputs, model: LM, num_rollouts=None, isgreedy=True, **kwargs):
    results = []
    
    if len(inputs)==1:
        completions, temperature = model.generate(inputs, num_rollouts, isgreedy, **kwargs)
        results = completions
    else: 
        for inp in inputs:
            completions, temperature = model.generate(inp, num_rollouts, isgreedy, **kwargs)
            results.append(completions) 
    return results


lm_tokenizer, lm_model, post_tokenizer, post_model = None, None, None, None
def get_model_output(
    template_role: str,
    system_prompt: Optional[str],
    template_inputs: List[List[Any]],
    tokenizer: Any,
    model: Any,
    prompt_suffix: str,
    # NOTE: use_fewshot is an unused parameter in this function.
    use_fewshot: bool,
    num_rollouts: int,
    is_greedy: bool = True,
    **kwargs,
) -> Tuple[Any, List[str]]:
    """
    Constructs prompts from templates and submits them to the model for generation.

    Args:
        template_role: The key for the desired prompt template.
        system_prompt: An optional system message to guide the model's behavior.
        template_inputs: A list of lists, where each inner list contains the arguments for a prompt template.
        tokenizer: The model's tokenizer.
        model: The language model instance.
        prompt_suffix: A string to append to each prompt after formatting.
        use_fewshot: (Unused) A flag that was likely intended for few-shot prompting.
        num_rollouts: The number of sequences to generate for each prompt.
        is_greedy: A flag to control the decoding strategy.
        **kwargs: Additional arguments passed to the generation function.

    Returns:
        A tuple containing the Ray object for the asynchronous generation task
        and the list of fully constructed prompts sent to the model.
    """
    template = templates[template_role]
    
    # Format each input using the specified template.
    formatted_queries = [template.format(*inp) for inp in template_inputs]
    
    prompts = []
    for query in formatted_queries:
        # Create the standard message format for model interaction.
        messages = [{"role": "user", "content": query}]
        
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # Construct the final prompt string.
        prompt = make_prompt(tokenizer, messages)
        prompt += prompt_suffix
        prompts.append(prompt)
    # if kwargs.get('log',False):
    #     print(prompts)
    # Launch the remote generation task using Ray.
    generation_task = generate.remote(prompts, model, num_rollouts, isgreedy=is_greedy, **kwargs)
    return generation_task, prompts

class RefinementProcessor:
    """Handles the iterative refinement process for a single generated response."""

    def __init__(self, node: Any, tokenizer: Any, model: Any, post_tokenizer: Any, post_model: Any, stop_threshold: float, max_steps: int, num_expansion: int = 2):
        self.node = node
        self.tokenizer = tokenizer
        self.model = model
        self.post_tokenizer = post_tokenizer
        self.post_model = post_model
        self.stop_threshold = stop_threshold
        self.max_steps = max_steps
        self.num_expansion = num_expansion

    def run(self, initial_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full iterative refinement loop on a generated "thinking" process.

        Args:
            initial_info: A dictionary containing the initial model rollout and its metadata.

        Returns:
            An updated info dictionary containing the results of the refinement process.
        """
        # The 'initial_ppl' is a tuple where the first element is the log perplexity.
        initial_perplexity = initial_info.get('initial_ppl', (1000.0,))
        print(f"===> Initial PPL: {initial_perplexity[0]}")
        if initial_perplexity[0] < self.stop_threshold:
            print("===> Skipping refinement due to low initial PPL.")
            return initial_info

        thinking_segments = initial_info.get('thinking_segments', [])
        if not thinking_segments:
            return initial_info

        print(f"====> Starting refinement on {len(thinking_segments)} thinking steps.")
        
        finalized_thinking_steps = []
        best_ppl_so_far = initial_perplexity
        
        # Iterate through each segment of the thinking process to refine it.
        for i in range(min(self.max_steps, len(thinking_segments))):
            # Reconstruct the thinking process parts: before, current, and after the segment being refined.
            before_segment = "\n\n".join(finalized_thinking_steps)
            segment_to_refine = thinking_segments[i].strip()
            after_segments = "\n\n".join(s.strip() for s in thinking_segments[i+1:])
            
            # Generate and evaluate several possible refinements for the current segment.
            best_candidate, best_ppl, all_candidates = self._refine_one_step(
                before_segment, segment_to_refine, after_segments
            )
            
            # Decide whether to keep the original segment or use a generated refinement.
            choice = 'original'
            chosen_text = segment_to_refine
            if best_candidate is not None and best_ppl[0] < best_ppl_so_far[0]:
                best_ppl_so_far = best_ppl
                chosen_text = best_candidate['refinement']
                choice = f"refinement_No.{best_candidate['id']}"

            finalized_thinking_steps.append(chosen_text)
            
            # Log the details of this refinement step.
            initial_info[f"refine_thinking_step_No.{i+1}"] = {
                'segment_to_refine': segment_to_refine,
                'choice': choice,
                'chosen_refinement': chosen_text,
                'after_avg_token_logp': best_ppl_so_far[0],
                'possible_refinements': all_candidates,
            }

            if best_ppl_so_far[0] < self.stop_threshold:
                print("===> Stopping refinement early as PPL threshold reached.")
                break

        if len(finalized_thinking_steps)<len(thinking_segments):
            # Append the rest of the original segments without refinement.
            finalized_thinking_steps.extend(s.strip() for s in thinking_segments[i+1:])
        
        initial_info['corrected_thinking'] = finalized_thinking_steps
        early_terminate = i+1<len(thinking_segments)
        initial_info['finalized_avg_token_logp'] = best_ppl_so_far[0]
        # If any refinement was done, perform one final rollout with the complete refined thinking.
        # if finalized_thinking_steps and early_terminate:
        #     new_thinking_prompt = "<think>\n" + "\n\n".join(finalized_thinking_steps)
        #     role_suffix = '_cn' if self.node.memory['is_chinese'] else '_en'
        #     rollout_texts, _, _, _ = direct_rollout(self.node, new_thinking_prompt, 1, role="standard_inference"+role_suffix)
        #     initial_info['finalized_response'] = rollout_texts[0] if len(rollout_texts) else None

        return initial_info

    def _refine_one_step(self, before: str, current: str, after: str) -> Tuple[Optional[Dict], Tuple[float, List], List[Dict]]:
        """Generates and evaluates possible refinements for a single thinking segment."""
        num_samples = self.num_expansion
        # The arguments are: question, reference_answer, text_before, text_to_replace, text_after, original_text
        prompt_args = (self.node.memory['q'], self.node.memory['ref'], before, current, after, current)
        role_suffix = '_cn' if self.node.memory['is_chinese'] else '_en'
        pre_trigger = "Let's find out what can be improved and enriched to better align with the target text.\n" if role_suffix=='_en' else "让我看看这段有什么可以做修改、优化、补充的地方，从而更加贴合目标文本\n"
        rollouts_obj, _ = get_model_output(
            template_role='segment_edits'+role_suffix, 
            system_prompt=default_sys, 
            template_inputs=[prompt_args], 
            tokenizer=self.tokenizer, 
            model=self.model, 
            prompt_suffix=pre_trigger+"<analyze>\n",
            use_fewshot=False,
            num_rollouts=num_samples, 
            is_greedy=False
        )
        # The return from ray.get should be (texts, logps, offsets), but only texts are used here.
        rollout_texts, _, _ = ray.get(rollouts_obj)
        del rollouts_obj

        refinement_candidates = []
        best_candidate_info = None
        # Initialize with a high perplexity value.
        best_perplexity = (float('inf'), [])
        if np.random.uniform()<0.5:
            eng_trigger = "Wait"
        elif np.random.uniform()<0.8:
            eng_trigger = "But wait"
        else:
            eng_trigger = "Meanwhile"
        trigger = eng_trigger if role_suffix == '_en' else "等等我再想想"
        for i, rollout_text in enumerate(rollout_texts):
            # if np.random.uniform()<0.25: # extra wait 
            #     temp = rollout_text.split('</refine>')[0].strip()
            #     rollouts_obj2, _ = get_model_output(
            #         template_role='segment_edits'+role_suffix, 
            #         system_prompt=default_sys, 
            #         template_inputs=[prompt_args], 
            #         tokenizer=self.tokenizer, 
            #         model=self.model, 
            #         prompt_suffix=pre_trigger+"<analyze>\n"+rollout_text+f"\n\n{trigger}",
            #         use_fewshot=False,
            #         num_rollouts=1, 
            #         is_greedy=False
            #     )
            #     # The return from ray.get should be (texts, logps, offsets), but only texts are used here.
            #     rollout_texts2, _, _ = ray.get(rollouts_obj2)
            #     del rollouts_obj2
            #     new_roll = f"{temp}\n\n{trigger}{rollout_texts2[0]}"
            #     rollout_text = new_roll

            # Extract the refined text from within the <refine> tags.
            last_block_start = rollout_text.rfind("<refine>")
            if last_block_start == -1:
                print("Warning: <refine> tag not found in output, skipping.")
                continue

            block_end = rollout_text.rfind("</refine>")
            start_pos = last_block_start + len("<refine>")
            refinement_text = rollout_text[start_pos:block_end if block_end != -1 else None].strip()
            
            # Create the full "thinking" process with the new refinement.
            recomposed_thinking = f"{refinement_text}"
            if before: recomposed_thinking = f"{before}\n\n{recomposed_thinking}"
            if after: recomposed_thinking = f"{recomposed_thinking}\n\n{after}"
            
            # Evaluate the new thinking process by calculating the perplexity of the reference answer.
            manager = PosteriorManager('standard_inference'+role_suffix, default_sys, [[self.node.memory['q']]], self.post_tokenizer, self.post_model, self.node.memory['ref'])
            _, posterior_prefix = manager.prepare(think_prefix, recomposed_thinking, "")
            ppl_obj, _ = manager.submit(posterior_prefix)
            # The 'compute' method returns a tuple (log_perplexity, debug_info).
            new_perplexity_result = manager.compute(ppl_obj)

            print(f"Refinement candidate {i} | New log PPL: {new_perplexity_result[0]}")
            candidate_info = {
                'id': i,
                'refinement': refinement_text,
                'generator': self.model.model_name,
                'raw_output': rollout_text,
                'raw_input_for_posterior': posterior_prefix,
                'avg_token_logp': new_perplexity_result[0],
            }
            refinement_candidates.append({f'expansion_No.{i}_of_segment': candidate_info})

            # If this candidate is better than the best one so far, update it.
            if new_perplexity_result[0] < best_perplexity[0]:
                best_perplexity = new_perplexity_result
                best_candidate_info = candidate_info
                
        return best_candidate_info, best_perplexity, refinement_candidates
    
class PosteriorManager:
    """Calculates the log probability of a reference answer given a thinking process."""
    def __init__(self, role: str, system_prompt: str, inputs: List[List[Any]], tokenizer: Any, model: Any, ref_answer: str):
        self.role = role
        self.system_prompt = system_prompt
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.model = model
        self.ref_answer = ref_answer
        _, _, answer_steps = breakdown_steps(ref_answer)
        self.answer_prefix = answer_steps[0]
        self.pred_answer = "".join(answer_steps[1:])
        
    def prepare(self, think_prefix: str, thinking_process: str, ref_answer: str = None) -> Tuple[str, str]:
        """Prepares the prompt for posterior probability calculation."""
        # The part of the prompt before the reference answer.
        self.prefix_before_answer = f"{think_prefix}{thinking_process}\n</think>\n<answer>\n{self.answer_prefix}"
        # The full prompt including the reference answer.
        
        self.posterior_prefix = f"{self.prefix_before_answer}{self.pred_answer}\n</answer>"
        return self.prefix_before_answer, self.posterior_prefix
    
    def submit(self, prefix: str) -> Tuple[Any, List[str]]:
        """Submits the prompt to the model to get token log probabilities."""
        self.rollouts_obj, self.real_input_prompts = get_model_output(
            self.role, self.system_prompt, self.inputs, self.tokenizer, self.model, prefix, 
            use_fewshot=False, num_rollouts=1, is_greedy=False, prompt_only=True
        )
        return self.rollouts_obj, self.real_input_prompts

    def compute(self, rollouts_obj: Any) -> Tuple[float, List[Dict]]:    
        """Computes the log perplexity of the reference answer from the model's output."""
        _, prompt_logprobs, _ = ray.get(rollouts_obj)
        if not prompt_logprobs: return 1000.0, []
        
        current_prompt_logprobs = prompt_logprobs[0]
        
        # This complex logic is used to find the exact tokens corresponding to the reference answer.
        offsets = self.tokenizer(self.posterior_prefix, return_offsets_mapping=True).offset_mapping 
        start_char_index = len(self.prefix_before_answer)
        end_char_index = start_char_index + len(self.pred_answer)

        # Find the number of tokens to count backward from the end of the prompt to find the answer's start.
        tokens_to_go_back = 0
        for i, (start, end) in enumerate(offsets[::-1]):
            if start <= start_char_index:
                tokens_to_go_back = i + 1
                break
        
        # Find how many tokens the reference answer spans.
        answer_token_span = 0
        for j, (start, end) in enumerate(offsets[-tokens_to_go_back:]):
            if end >= end_char_index:
                answer_token_span = j + 1
                break
        
        answer_logps = []
        # Sum the log probabilities of the tokens that make up the reference answer.
        logprob_slice = current_prompt_logprobs[-tokens_to_go_back : -tokens_to_go_back + answer_token_span]
        for logp_dict in logprob_slice: 
            # The dictionary may have multiple keys; the first one corresponds to the prompt token.
            major_key = list(logp_dict)[0]
            logp = logp_dict[major_key]['logprob']
            answer_logps.append(np.clip(logp, -2.0, 0.0))
            
        # Calculate the negative mean log probability (log perplexity). A lower value is better.
        log_perplexity = -np.mean(answer_logps) if answer_logps else 1000.0
        
        # For debugging, gather the log probability info for tokens surrounding the answer.
        debug_slice = current_prompt_logprobs[-tokens_to_go_back-2 : -tokens_to_go_back + answer_token_span + 2]
        debug_logprob_info = [logp_dict[list(logp_dict)[0]] for logp_dict in debug_slice]
        return log_perplexity, debug_logprob_info
    
def direct_rollout(node, prefix, n_sample=1, role=None, log=False):
    
    role, sysp, in_keys = ('standard_inference' if role is None else role), default_sys, ['q', 'ref'] 
    node.memory['gen_role'] = role
    
    inputs = [node.memory[k] for k in in_keys]
    tok, model = lm_tokenizer, lm_model
    node.memory['generator'] = model.model_name
    inp = [inputs]
    # print(f"===> {role}: {n_sample} for {len(inp)} queries")
    rollouts_obj, real_input_prompts = get_model_output(role, sysp, inp, tok, model, prefix, False, n_sample, is_greedy=False)
    rollout_texts, rollout_logps, rollout_offsets = ray.get(rollouts_obj)
    del rollouts_obj
    return rollout_texts, rollout_logps, rollout_offsets, real_input_prompts



@ray.remote
def process_item(
    item: Dict[str, Any],
    file_prefix: str,
    rank: int,
    n_sample: int, 
    configs=dict()
):
    uid = item['extra_info']['index']
    output_fname = f"{file_prefix}_{uid}_rk{rank}"
    # if glob(f"{output_fname}*"):
    # for fp in glob(f"{file_prefix}_{uid}*"):
    for fp in glob(f"{output_fname}*"):
        try: 
            tmp = json.load(open(fp))
            if "alist" in tmp:
                print(f"Skipping existing item: {uid}")
                return True
        except Exception as e:
            print(e)
            print(f'wrong loading {fp}')
            continue
    
    stop_thresh = configs['processing']['stop_thresh']
    max_step = configs['processing']['max_step']
    num_expansion = configs['processing']['num_expansion']
    q = item['question']
    is_chinese = contains_chinese(q)
    has_think = "</think>" in item['solution']
    if has_think:
        ref = item['solution'].split('</think>')[-1].strip() 
    else:
        ref = item['solution']
    node = Node(ref=ref, raw_q=item['question'], info={'uid': item['extra_info']['index'], 'old_solution': item['solution'] if has_think else None, 'is_chinese': is_chinese})
    node.memory.update({k:v for k,v in item.items() if not isinstance(v, np.ndarray) if k not in {'solution'}})
    
    fname = output_fname

    tok, model = lm_tokenizer, lm_model

    # 1. Initial Rollout
    # n_sample = 1
    generation_role = 'initial_thinking_'+('cn' if is_chinese else 'en')
    if is_chinese:
        if np.random.uniform()>0.5: pre_trigger = "好的"
        else: pre_trigger = "嗯"
    else:
        if np.random.uniform()>0.8: pre_trigger = "Okay, I am given"
        elif np.random.uniform()>0.4: pre_trigger = "Alright, the user"
        else: pre_trigger = "Alright"
    rollout_texts, rollout_logps, rollout_offsets, real_input_prompts = direct_rollout(node, "<think>\n"+pre_trigger, n_sample, role=generation_role, log=True)
    # print(rollout_texts[0])
    outcomes = []
    expanded_prompts = [pp for pp in real_input_prompts for _ in range(n_sample)]
    flag = True
    for roll, inpprompt, token_logps, token_to_text_offsets in zip(rollout_texts, expanded_prompts, rollout_logps, rollout_offsets):
        ntoken = len(token_logps)
        if ntoken<100: 
            print("num token too short, skip")
            return False
        info = dict(ntokens=ntoken,)
        
        # separate thinking and code answer
        roll = pre_trigger + roll
        thinking = roll.split('</think>')[0]
        answer_code = roll.split('<answer>')[-1].split('</answer>')[-1].strip()

        # breakdown thinking to steps 
        aa,bb,thinking_segments = breakdown_steps(thinking)
        if len(thinking_segments)==1:
            return False
        # get PPL of initial rollout
        role, sysp = 'standard_inference_'+('cn' if is_chinese else 'en'), default_sys 
        inp = [[node.memory['q']]]
        manager = PosteriorManager(role, sysp, inp, post_tokenizer, post_model, node.memory['ref'])

        posterior_prefix1, posterior_prefix = manager.prepare(think_prefix, thinking)
        rollouts_obj, real_input_prompts = manager.submit(posterior_prefix)
        noreplace_log_ppl = manager.compute(rollouts_obj)
        ppl = noreplace_log_ppl
        if ppl[0]<stop_thresh:
            flag = False
            print('early stopping')

            
        # save info
        info.update(dict(initial_response=roll, 
                            answer=answer_code,
                            initial_ppl=ppl, 
                            generator=model.model_name,
                            thinking_segments=thinking_segments,
                        #  rawinput=inpprompt, 
                    ))
    
        outcomes.append(info)

    if flag: 
        # 2. iterative refinement for each response
        refinement_processor = RefinementProcessor(
            node, tok, model, post_tokenizer, post_model,
            stop_threshold=stop_thresh, 
            max_steps=max_step,
            num_expansion=num_expansion
        )
        
        updated_results = []
        for info in outcomes:
            refined_info = refinement_processor.run(info)
            updated_results.append(refined_info if refined_info is not None else info)
        
        node.memory['alist'] = updated_results
    
        flag = False 
    else:
        node.memory['alist'] = outcomes

    result_file = f"{fname}_meta.jsonl"
    with open(result_file, "w") as file:
        line = json.dumps(node.memory,indent=2,ensure_ascii=False)

        file.write(line + "\n")
    print(f'dumped to {result_file}')
      
    return flag
        

def main(args: argparse.Namespace):
    """Main function to load data, initialize models, and run the processing loop."""
    config = load_config(args.config_file)
    file_prefix = config["output"]["file_prefix"]
    folder = os.path.sep.join(file_prefix.split(os.path.sep)[:-1])
    os.makedirs(folder, exist_ok=True)
    num_rollouts = config["processing"]["num_rollouts"]

    # Create output directory
    os.makedirs(os.path.dirname(config["output"]["file_prefix"]), exist_ok=True)
    
    # Initialize Ray
    ray.init(num_cpus=args.num_cpus)

    # Load models and tokenizers
    print("Loading models and tokenizers...")
    global lm_tokenizer, lm_model, post_model, post_tokenizer
    config["model"]["model_args"]["port"] = args.port
    config["judge_model"]["model_args"]["port"] = f"2{args.port}"
    config["model"]["model_name"] = args.model 
    if args.posterior_model!="":
        config["judge_model"]["model_name"] = args.posterior_model 
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["model"]["model_name"],
        use_fast=True,
        padding_side="right",
        truncation_side="right",
    )
    lm_model = LM(
        model_type=config["model"]["model_type"],
        model_name=config["model"]["model_name"],
        model_url=None, # lm_url,
        num_rollouts=num_rollouts,
        tokenizer=lm_tokenizer,
        **config["model"]["model_args"],
    )
    if config["judge_model"]["use"]:
        post_tokenizer = transformers.AutoTokenizer.from_pretrained(
            config["judge_model"]["model_name"],
            use_fast=True,
            padding_side="right",
            truncation_side="right",
        )
        post_model = LM(
            model_type=config["judge_model"]["model_type"],
            model_name=config["judge_model"]["model_name"],
            model_url=None, # lm_url,
            num_rollouts=num_rollouts,
            tokenizer=lm_tokenizer,
            **config["judge_model"]["model_args"],
        )
    else:
        post_tokenizer, post_model = lm_tokenizer, lm_model
    path = config['input']['file_path']
    if path.endswith("parquet"):
        df = pd.read_parquet(path)
        data = df.to_dict('records')
        
    else:
        data = json.load(open(path))
        for entry in data:
            entry['extra_info'] = {'index': entry['index']}
    
    num_each = len(data) // args.total_ranks + 1
    rank = args.rank%args.total_ranks
    rank_data = data[rank * num_each : (rank + 1) * num_each]
    # np.random.shuffle(rank_data)
    print(f"Rank {args.rank}/{args.total_ranks}: Processing {len(rank_data)} items.")
    st = time.time()
    num_para = 400
    for idx in range(0, len(rank_data), num_para):
        print(f"starting batch: {idx} ")
        rank_data_ = rank_data[idx:idx+num_para]
        task_futures = [
            process_item.remote(
                    item,
                    file_prefix,
                    args.rank,
                    num_rollouts,
                    configs=config
                )
            for item in rank_data_
        ]

        # Wait for all tasks to complete
        ray.get(task_futures)
        print(f"Total time elapsed for {num_para} queries: {(time.time() - st) / 60:.2f} minutes.")
    
    ray.shutdown()
    
    logging.info("Finished processing the JSON file.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run language model generation and refinement pipeline.")
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--model", type=str, required=True, help="Path to generator.")
    parser.add_argument("--posterior_model", type=str, default="", help="Path to generator.")
    parser.add_argument("--port", type=int, required=True, help="Port.")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current process.")
    parser.add_argument("--total-ranks", type=int, required=True, help="Total number of parallel processes.")
    parser.add_argument("--num-cpus", type=int, default=32, help="Number of CPUs to allocate for Ray.")
    
    args = parser.parse_args()
    main(args)
