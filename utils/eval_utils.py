import copy, re, openai, os, time
import numpy as np
import warnings
from openai import OpenAI
from google import genai
from google.genai import types
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_judge(model_name, implementation="api"):
    assert implementation in ['api', 'huggingface']
    if implementation=='huggingface':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif implementation=='api':
        model = model_name
        tokenizer = None
    return model, tokenizer

def get_judge_prompt(question, response, ground_truth):
    judge_prompt = f"""You will be given a question, a model response and a ground-truth answer. Your task is to determine whether the model response is correct based on the ground-truth answer. The model response should contain all information in the ground-truth answer.

Question: {question}

Model Response: {response}

Ground-Truth Answer: {ground_truth}

Directly output "Correct" or "Incorrect":
"""
    return judge_prompt

def llm_judge_api(question, response, ground_truth, judge_model):
    judge_prompt = get_judge_prompt(question, response, ground_truth)

    contents = [
        {"type": "text", "text": judge_prompt}
    ]
    judge_response, _ = test_chat_completion_openai(judge_model, contents, max_tokens=16)

    return judge_response == "Correct"

def llm_judge_hf(question, response, ground_truth, judge_model, tokenizer):
    judge_prompt = get_judge_prompt(question, response, ground_truth)

    judge_response = test_chat_completion_hf(judge_model, judge_prompt, tokenizer, max_tokens=4)

    return judge_response == "Correct"

def test_chat_completion_gemini(model, question, video_path, max_new_tokens=1024, max_try=5, temperature=0., thinking_budget=8192):
    api_key=os.environ.get("GEMINI_API_KEY")
    assert api_key, "Please set GEMINI_API_KEY in environment!"
    client = genai.Client(
        api_key=api_key,
    )
    max_try_upload = 5
    while True:
        try:
            print("Uploading file...")
            video_file = client.files.upload(file=video_path)
            print(f"Completed upload: {video_file.uri}")

            # Check whether the file is ready to be used.
            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(1)
                video_file = client.files.get(name=video_file.name)
            break
        except Exception as e:
            print(f"Error during file upload: {e}, {max_try_upload} retries remaining...")
            dr = client.files.delete(name=video_file.name)
            max_try_upload -= 1
            if max_try_upload <= 0:
                return "", None
            time.sleep(10)
    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    print('Done')
    contents = [video_file, question]

    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=None,
        top_k=None,
        max_output_tokens=max_new_tokens,
        response_mime_type="text/plain",
        thinking_config=None if model=="gemini-2.0-flash" else types.ThinkingConfig(thinking_budget=thinking_budget)
    )

    while True:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config
            )
            print(response)
            token_count = {"candidates_token_count": response.usage_metadata.candidates_token_count, \
                                    "prompt_token_count": response.usage_metadata.prompt_token_count, \
                                    "thoughts_token_count": response.usage_metadata.thoughts_token_count , \
                                    "total_token_count": response.usage_metadata.total_token_count}
            return response.text, token_count

        except Exception as e:
            print(f"Error during generate_content: {e}")
            max_try -= 1
            if max_try <= 0:
                return "", None
            print(f"Exception occurred. {max_try} retries remaining...")
            time.sleep(10)

def test_chat_completion_hf(model, prompt, tokenizer, max_tokens=1024):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=0.,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text

def test_chat_completion_openai(model, contents, max_tokens=1024, max_try=5, temperature=0.):

    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in environment!"
    base_url = os.environ.get("OPENAI_API_BASE")
    assert base_url, "Please set OPENAI_API_BASE in environment!"

    client = OpenAI(
        api_key=api_key, base_url=base_url
    )

    while True:
        try:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": contents,
                    }
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = completion.choices[0].message.content
            token_count = {"completion_tokens": completion.usage.completion_tokens, "prompt_tokens": completion.usage.prompt_tokens, "total_tokens": completion.usage.total_tokens}
            return response, token_count
        except Exception as e:
            print(f"Error during generate_content: {e}")
            max_try -= 1
            if max_try <= 0:
                return "", None
            print(f"Exception occurred. {max_try} retries remaining...")
            time.sleep(60)

def extract_final_answer(response):
    if not response:
        return False
    for think_tok in ["\u25c1/think\u25b7"]:
        response = response.split(think_tok)[1] if think_tok in response else response
    match = re.search(r"Final Answer\s*(.*?)\s*$", response, re.DOTALL)
    if match:
        response = match.group(1)
        return response
    else:
        print("Final Answer not found.")
        return False

def find_blank(board, board_size):
    """返回空格（0）在 board 中的行、列索引"""
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                return i, j
    return None

def get_next_board_grid(move, board, coord):
    # get next coord
    x, y = copy.deepcopy(coord)
    board = copy.deepcopy(board)
    if move=='x-1':
        x = x-1
    elif move=='x+1':
        x = x+1
    elif move=='y-1':
        y = y-1
    elif move=='y+1':
        y = y+1

    if not (0 <= x < len(board) and 0 <= y < len(board)):
        print(f"警告：尝试在边界外更新网格状态 ({x}, {y})。")
        return False, coord

    # Flip target cell
    board[y][x] = 1 - board[y][x]

    # Flip neighbors
    neighbors_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right
    for dx, dy in neighbors_offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(board) and 0 <= ny < len(board): # Check bounds
            board[ny][nx] = 1 - board[ny][nx]
    return board, (x, y)

def get_next_board_hrd(move, board):
        bi, bj = find_blank(board, len(board))
        if move == "up":
            target = (bi - 1, bj)
        elif move == "down":
            target = (bi + 1, bj)
        elif move == "left":
            target = (bi, bj + 1)
        elif move == "right":
            target = (bi, bj - 1)
        else:
            print("Unknown Move:" + move)
            return board, False
        
        ti, tj = target
        if ti < 0 or ti >= len(board) or tj < 0 or tj >= len(board):
            return board, False
        new_board = copy.deepcopy(board)
        new_board[bi][bj], new_board[ti][tj] = new_board[ti][tj], new_board[bi][bj]

        return new_board, True

def get_next_board_cup(move, board):

    coordmap = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4,
    }

    try:
        if not isinstance(move, list):
            warnings.warn("Move must be a list", UserWarning)

        if len(move) != 2 or len(move[0]) != 2 or len(move[1]) != 2:
            warnings.warn("Invalid move format", UserWarning)
        r1, c1 = coordmap.get(move[0][0], None), coordmap.get(move[0][1], None)
        r2, c2 = coordmap.get(move[1][0], None), coordmap.get(move[1][1], None)

        if None in (r1, c1, r2, c2):
            warnings.warn("Invalid coordinate mapping", UserWarning)

        if (not (0 <= r1 < len(board)) or 
            not (0 <= c1 < len(board[0])) or
            not (0 <= r2 < len(board)) or 
            not (0 <= c2 < len(board[0]))):
            warnings.warn("Board index out of range", UserWarning)
        new_board = copy.deepcopy(board)
        new_board[r1][c1], new_board[r2][c2] = new_board[r2][c2], new_board[r1][c1]
    except Exception as e:
        warnings.warn(f"Unexpected error: {str(e)}", UserWarning)
        new_board = copy.deepcopy(board)  
        print("move:\n", move)
    return new_board

def get_next_files(commands, all_states, tgt_path):
    states = {s: set(list(all_states[s].values())[-1]) for s in all_states}
    path_pattern = r'path\d+'
    for cmd in commands:
        files = set(re.search(r'\{([^}]+)\}', cmd).group(1).split(','))
        paths = re.findall(path_pattern, cmd)
        if cmd.startswith('rm'):
            states[paths[0]] = states[paths[0]] - files
        elif cmd.startswith('touch'):
            states[paths[0]] = states[paths[0]] | files
        elif cmd.startswith('cp'):
            files = files & states[paths[0]]    # only files in path[0] can be copied to path[1]
            states[paths[1]] = states[paths[1]] | files
        elif cmd.startswith('mv'):
            files = files & states[paths[0]]    # only files in path[0] can be moved to path[1]
            states[paths[0]] = states[paths[0]] - files
            states[paths[1]] = states[paths[1]] | files
        for s in all_states:
            all_states[s][cmd] = copy.deepcopy(states[s])
    pred_files = list(all_states[tgt_path].values())[-1]
    return pred_files

def extract_move_hrd(response, model_path, tokenizer, judge_implement="api"):
    prompt = f"""You will be given a model-generated response describing a sequence of movements. Your task is to extract the movements in the order they appear and return them as a list (e.g., ['left', 'up', 'down', 'right']).

Model Response: {response}

Extracted Movements:
"""
    contents = [
        {"type": "text", "text": prompt}
    ]
    if judge_implement=="api":
        moves, _ = test_chat_completion_openai(model_path, contents, max_tokens=512)
    elif judge_implement=="huggingface":
        moves = test_chat_completion_hf(model_path, prompt, tokenizer, max_tokens=512)
    try:
        moves = [m.lower() for m in eval(moves)]
        return moves
    except SyntaxError:
        return False

def extract_move_cup(response, model_path, tokenizer, judge_implement="api"):
    prompt = f"""You will be given a model-generated response describing a sequence of cup swaps. Each swap is represented as a pair of coordinates—for example, (a1, b2)—indicating the two positions being swapped. 

Your task:
Extract all coordinate pairs from the response in the exact order they appear, and return them as a list of tuples.

Format your answer like this:
[('a1', 'b2'), ('c1', 'b1'), ('a3', 'b2')]

Model Response:
{response}

Extracted Swaps:
"""
    contents = [
        {"type": "text", "text": prompt}
    ]
    if judge_implement=="api":
        moves, _ = test_chat_completion_openai(model_path, contents, max_tokens=512)
    elif judge_implement=="huggingface":
        moves = test_chat_completion_hf(model_path, prompt, tokenizer, max_tokens=512)
    try:
        moves = eval(moves)
        return moves
    except:
        print(f"Fail to extract moves: {moves}")
        return False

def extract_file_cmd(response, model_path, tokenizer, judge_implement="api"):
    prompt = f"""You will be given a model-generated response regarding a file operation command in Linux system.

Your task:
Identify and extract only the actual command from the model response, removing any irrelevant or descriptive text.

Model Response:
{response}

Extracted Command:
"""
    contents = [
        {"type": "text", "text": prompt}
    ]
    if judge_implement=="api":
        moves, _ = test_chat_completion_openai(model_path, contents, max_tokens=512)
    elif judge_implement=="huggingface":
        moves = test_chat_completion_hf(model_path, prompt, tokenizer, max_tokens=512)
    moves = moves.strip('`').replace('bash\n', '').split(' & ')
    return moves

def extract_move_card(response, model_path, tokenizer, judge_implement="api"):
    prompt = f"""You will be given a model-generated response describing a sequence of operations performed to cards. Each operation either adds or removes a card from pile0 or pile1.

Your task:
- Extract all valid operations and return them as a list of strings.

- Each operation must involve either adding or removing a card to or from pile0 or pile1.

- If no valid operations are found, return an empty list ([]).

Format your answer like this:
['add 6 of Hearts to pile0', 'remove King of Clubs from pile0']

Model Response:
{response}

Extracted Operations:
"""
    contents = [
        {"type": "text", "text": prompt}
    ]
    if judge_implement=="api":
        moves, _ = test_chat_completion_openai(model_path, contents, max_tokens=512)
    elif judge_implement=="huggingface":
        moves = test_chat_completion_hf(model_path, prompt, tokenizer, max_tokens=512)
    try:
        moves = eval(moves)
        return moves
    except:
        print(f"Fail to extract moves: {moves}")
        return False

def extract_move_chip(response, model_path, tokenizer, judge_implement="api"):
    prompt = f"""You will be given a model-generated response describing a sequence of operations involving chips and cups. Each operation either adds or removes a chip from cup0 or cup1.

Your task:
- Extract all valid operations and return them as a list of strings.

- Each operation must involve either adding or removing a chip to or from cup0 or cup1.

- If no valid operations are found, return an empty list ([]).


Format your answer like this:
['add 20 to cup0', 'remove 50 cup0']

Model Response:
{response}

Extracted Operations:
"""
    contents = [
        {"type": "text", "text": prompt}
    ]
    if judge_implement=="api":
        moves, _ = test_chat_completion_openai(model_path, contents, max_tokens=512)
    elif judge_implement=="huggingface":
        moves = test_chat_completion_hf(model_path, prompt, tokenizer, max_tokens=512)
    try:
        moves = eval(moves)
        return moves
    except:
        print(f"Fail to extract moves: {moves}")
        return False

def extract_tgt_board_grid(question):
    match = re.search(r"the desired arrangement of black and white pieces:\s*`([^`]+)`", question)
    if match:
        arrangements = match.group(1)
    else:
        raise ValueError("No target board found.")
    match = re.search(r"presents a (\d+)x(\d+) grid", question)
    if match:
        num_row, num_col = int(match.group(1)), int(match.group(2))
    else:
        raise ValueError("No board size found.")
    
    coordmap = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
    }

    board = np.zeros((num_row, num_col), dtype=int)
    for argm in arrangements.split(', '):
        coord, color = argm.split(': ')
        coord = coord.strip('()').split(',')
        x, y = coordmap[coord[0]], coordmap[coord[1]]
        board[x][y] = 1 if color=='black' else 0
    return board

def extract_tgt_board_hrd(question):
    match = re.search(r"the desired number arrangement:\s*`([^`]+)`", question)
    if match:
        arrangements = match.group(1)
    else:
        raise ValueError("No target board found.")
    match = re.search(r"on a (\d+)x(\d+) board", question)
    if match:
        num_row, num_col = int(match.group(1)), int(match.group(2))
    else:
        raise ValueError("No board size found.")
    
    coordmap = {
        'a': num_row - 1,
        'b': num_row - 2,
        'c': num_row - 3,
        'd': num_row - 4,
        'e': num_row - 5,
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4,
    }

    board = np.zeros((num_row, num_col), dtype=int)
    for argm in arrangements.split(', '):
        coord, num = argm.split(': ')
        coord = coord.strip('()').split(',')
        x, y = coordmap[coord[0]], coordmap[coord[1]]
        board[x][y] = int(num)
    return board

def extract_tgt_board_cup(question):
    match = re.search(r"achieve the desired distribution of coins:\s*`([^`]+)`", question)
    if match:
        coords = match.group(1)
    else:
        raise ValueError("No target board found.")
    match = re.search(r"puzzle on a (\d+)x(\d+) board", question)
    if match:
        num_row, num_col = int(match.group(1)), int(match.group(2))
    else:
        raise ValueError("No board size found.")
    
    coordmap = {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 3,
        '5': 4,
    }

    board = np.zeros((num_row, num_col), dtype=int)
    for coord in coords.split(', '):
        x, y = coordmap[coord[0]], coordmap[coord[1]]
        board[x][y] = 1
    return board

def extract_tgt_files(question):
    match = re.search(r"contains exactly the following files:\s*`([^`]+)`", question)
    if match:
        files = match.group(1)
    else:
        raise ValueError("No target files found.")
    files = set(files.split(', '))
    match = re.search(r"to ensure that\s*`([^`]+)` contains exactly", question)
    if match:
        path = match.group(1)
    else:
        raise ValueError("No board size found.")
    return path, files

def extract_tgt_cards(question):
    match = re.search(r"contains exactly the following cards from top to bottom:\s*`([^`]+)`", question)
    if match:
        cards = match.group(1)
    else:
        raise ValueError("No target cards found.")
    cards = cards.split(', ')
    match = re.search(r"to ensure that\s*`([^`]+)` contains exactly", question)
    if match:
        pile = match.group(1)
    else:
        raise ValueError("No board size found.")
    return pile, cards

def extract_tgt_chips(question):
    match = re.search(r"contains exactly the following chips:\s*`([^`]+)`", question)
    if match:
        chips = match.group(1)
    else:
        raise ValueError("No target cards found.")
    chips = chips.split(', ')
    chips = [int(chip) for chip in chips]
    match = re.search(r"to ensure that\s*`([^`]+)` contains exactly", question)
    if match:
        cup = match.group(1)
    else:
        raise ValueError("No board size found.")
    return cup, chips

def eval_op_hrd(response, question, src_board, model_path, tokenizer, judge_implement="api"):
    moves = extract_move_hrd(response, model_path, tokenizer, judge_implement)
    if not moves:
        return False
    tgt_board = extract_tgt_board_hrd(question)
    next_board = copy.deepcopy(src_board)
    for move in moves:
        next_board, is_valid = get_next_board_hrd(move, next_board)
        if not is_valid:
            continue
    return np.array_equal(np.array(next_board), tgt_board)

def eval_op_grid(response, question, src_board, coord, model_path, tokenizer, judge_implement="api"):

    movemap = {'left': 'x-1', 'right': 'x+1', 'up': 'y-1', 'down': 'y+1'}

    moves = extract_move_hrd(response, model_path, tokenizer, judge_implement)
    if not moves:
        return False
    if not all([m in movemap for m in moves]):
        return False
    moves = [movemap[m] for m in moves]

    tgt_board = extract_tgt_board_grid(question)
    boards = [copy.deepcopy(src_board)]
    for move in moves:
        next_board, coord = get_next_board_grid(move, boards[-1], coord)
        if not next_board:
            continue
        boards.append(next_board)
    return np.array_equal(np.array(boards[-1]), tgt_board)

def eval_op_cup(response, question, src_board, model_path, tokenizer, judge_implement="api"):
    moves = extract_move_cup(response, model_path, tokenizer, judge_implement)
    if not moves:
        return False
    tgt_board = extract_tgt_board_cup(question)
    next_board = copy.deepcopy(src_board)
    for move in moves:
        next_board = get_next_board_cup(move, next_board)
    return np.array_equal(np.array(next_board), tgt_board)

def eval_op_file(response, question, states, model_path, tokenizer, judge_implement="api"):
    cmds = extract_file_cmd(response, model_path, tokenizer, judge_implement)
    if (not cmds) or len(cmds)>2:                                               # at most two commands can be used
        return False
    cmd_files = {'rm': set(), 'touch': set()}
    for cmd in cmds:
        if (not cmd.startswith('rm -rf')) and (not cmd.startswith('touch')):    # only `rm -rf` and `touch` can be used
            return False
        match = re.search(r'\{([^}]+)\}', cmd)
        if not match:
            return False
        files = set(match.group(1).split(','))
        if cmd.startswith('rm -rf'):
            cmd_files['rm'] = files
        elif cmd.startswith('touch'):
            cmd_files['touch'] = files

    if cmd_files['rm'] & cmd_files['touch']:                                    # `touch` files and `rm -rf` files should not overlap
        print("Overlap")
        return False

    tgt_path, tgt_files = extract_tgt_files(question)
    pred_files = get_next_files(cmds, states, tgt_path)
    return pred_files == tgt_files

card_types = ['Hearts', 'Clubs', 'Diamonds', 'Spades']
card_ids = ['Ace'] + [str(i) for i in range(2, 11)] + ['Jack', 'Queen', 'King']
all_cards = set([f"{cid} of {c_type}" for cid in card_ids for c_type in card_types])

def eval_op_card(response, question, src_states, model_path, tokenizer, judge_implement="api"):
    moves = extract_move_card(response, model_path, tokenizer, judge_implement)
    if not moves:
        return False

    tgt_pile, tgt_cards = extract_tgt_cards(question)
    states = copy.deepcopy(src_states)
    for move in moves:
        if not move.startswith('add') and not move.startswith('remove'):
            return False
        if not move.endswith('pile0') and not move.endswith('pile1'):
            return False
        card = ' '.join(move.split()[1:-2])
        if card not in all_cards:
            return False
        action = move.split()[0]
        pile = move.split()[-1]
        if pile not in states:
            return False
        if action == 'add':
            states[pile].append(card)
        elif action == 'remove':
            if (not states[pile]) or (card != states[pile][0]):
                return False
            states[pile].remove(card)
        
    return states[tgt_pile] == tgt_cards

chip_types = [5, 10, 20, 50, 100]

def eval_op_chip(response, question, src_states, model_path, tokenizer, judge_implement="api"):
    moves = extract_move_chip(response, model_path, tokenizer, judge_implement)
    if not moves:
        return False

    tgt_cup, tgt_chips = extract_tgt_chips(question)
    states = copy.deepcopy(src_states)
    for move in moves:
        if not move.startswith('add') and not move.startswith('remove'):
            return False
        if not move.endswith('cup0') and not move.endswith('cup1'):
            return False
        chip = move.split()[1]
        if not chip.isdigit():
            return False
        chip = int(chip)
        if chip not in chip_types:
            return False
        action = move.split()[0]
        cup = move.split()[-1]
        if action == 'add':
            states[cup].append(chip)
        elif action == 'remove':
            if chip not in states[cup]:
                return False
            states[cup].remove(chip)
        
    return states[tgt_cup] == tgt_chips