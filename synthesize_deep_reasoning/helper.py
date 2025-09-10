import yaml

class Node:
    def __init__(self, ref, raw_q, gold=None, info=None):
        if info is not None:
            self.memory = info 
        else: self.memory = {}
        self.memory.update(dict(q=raw_q, ref=ref,history=[]))
        self.ref = ref
        self.raw_q = raw_q
        
def segment_response(inp, sep='\n\n'):
    segments = []
    temp = ''
    i = 0
    n = len(inp)
    
    while i < n:
        if inp[i:i+2] == '\\[':
            # Start of a LaTeX block, collect until we find '\\]'
            j = i + 2
            while j < n and inp[j:j+2] != '\\]':
                j += 1
            # Add the LaTeX block including '\\[' and '\\]'
            # segments.append(inp[i:j+2])
            temp += inp[i:j+2]
            i = j + 2  # Move past '\\]'
        elif inp[i:i+sep.count('\n')] == sep:
            segments.append(temp)
            temp = ''
            j = i+1
            while j<n and inp[j].strip()=='':
                j += 1
            segments.append(inp[i:j])
            i = j
            
        else:
            # Add character to current segment
            temp += inp[i]
            i += 1
    
    # If there's any remaining text, add it
    if temp:
        segments.append(temp)
    # return segments
    
    final = []
    buffer = []
    for seg in segments:
        if seg.strip()=='': 
            buffer.append(seg)
        elif len(seg)<100: # if previous is quite short, append with it
            buffer.append(seg)
        else:
            if len(buffer)>0: 
                prefix = "".join(buffer)
                buffer = []
                seg = prefix + seg 
            final.append(seg)
    if len(buffer)>0: 
        prefix = "".join(buffer)
        buffer = []
        if final: final[-1] += prefix
        else: final = [prefix]
    return final

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)



def make_prompt(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # prompt = prompt.split('<think>')[0]
    if prompt.strip().endswith("<think>"):
        prompt = prompt.strip()[:-len("<think>")] # split('<think>')[0]
    return prompt


def breakdown_steps(a):
    steps = segment_response(a)
    if len(steps)<=1:
        steps = segment_response(a, '\n')
    final = ""
    allsteps = []
    for idx, step in enumerate(steps):
        allsteps.append(f"<Step{idx+1}> {step}\n")
        final += allsteps[-1]
    return final, allsteps, steps


def segment_offsets(offsets, segments, logps):
    """
    Splits the 'offsets' list into sublists based on the cumulative sums of 'segments'.

    Args:
        offsets (list[int]): A sorted list of integers.
        segments (list[int]): A list of segment lengths.

    Returns:
        list[list[int]]: A list of lists, where each sublist contains offsets
                         belonging to the corresponding segment.
    """
    
    # Calculate the cumulative sum of segments to get the upper bounds.
    # The first element is the length of the first segment, the second is the
    # sum of the first two, and so on.
    segment_boundaries = np.cumsum(segments)
    
    # This will hold the final list of lists
    result = []
    
    # Pointer for the current position in the offsets list
    offset_idx = 0 
    
    # This will be the starting boundary for each segment's range.
    # It starts at 0 for the first segment.
    lower_bound = 0

    # Iterate through each segment's upper boundary
    for upper_bound in segment_boundaries:
        
        # This sublist will store the offsets for the current segment
        current_segment_offsets = []
        
        # Go through the offsets list starting from where we last left off
        while offset_idx < len(offsets) and offsets[offset_idx] < upper_bound:
            current_segment_offsets.append(offsets[offset_idx])
            offset_idx += 1
            
        result.append(current_segment_offsets)
        
        # The next segment's range will start from the end of the current one
        lower_bound = upper_bound
        
    token_belongs_to_segment = result 
    seg_logp_list = [] # list of list of logps for each segment
    cnt = 0
    for seg_i, token_included in enumerate(token_belongs_to_segment):
        num_tokens = len(token_included)
        assert num_tokens>0
        seg_logp_list.append(logps[cnt:cnt+num_tokens])
        cnt += num_tokens
    return seg_logp_list, token_belongs_to_segment


def equals(a, b):
    flag = a==b
    if not flag:
        flag = False 
        try: 
            xx = eval(a)
            yy = eval(str(b))
            flag = abs(xx-yy)<1e-4 
        except:
            pass 
        
    return flag 
