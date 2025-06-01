from ...smp import *
from .multiple_choice import extract_answer_from_item
from PIL import Image, ImageOps
import numpy as np
import re
import copy
import warnings

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

think_tokens = ["◁/think▷"]

def get_judge_prompt(question, response, ground_truth):
    judge_prompt = f"""You will be given a question, a model response and a ground-truth answer. Your task is to determine whether the model response is correct based on the ground-truth answer.

Question:
{question}

Model Response:
{response}

Ground-Truth Answer:
{ground_truth}

NOTE: The model response should contain all information in the ground-truth answer. Directly output "Correct" or "Incorrect":
"""
    return judge_prompt


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
    states = {s: set(all_states[s][-1]) for s in all_states}
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
            all_states[s].append(copy.deepcopy(states[s]))
    pred_files = all_states[tgt_path][-1]
    return pred_files

def extract_move_hrd(response, model):
    prompt = f"""You will be given a model-generated response describing a sequence of movements. Your task is to extract the movements in the order they appear and return them as a list (e.g., ['left', 'up', 'down', 'right']).

Model Response: {response}

Extracted Movements:
"""
    moves = model.generate(prompt, max_tokens=512)
    try:
        moves = [m.lower() for m in eval(moves)]
        return moves
    except SyntaxError:
        return False

def extract_move_cup(response, model):
    prompt = f"""You will be given a model-generated response describing a sequence of cup swaps. Each swap is represented as a pair of coordinates—for example, (a1, b2)—indicating the two positions being swapped. 

Your task:
Extract all coordinate pairs from the response in the exact order they appear, and return them as a list of tuples.

Format your answer like this:
[('a1', 'b2'), ('c1', 'b1'), ('a3', 'b2')]

Model Response:
{response}

Extracted Swaps:
"""
    moves = model.generate(prompt, max_tokens=512)
    try:
        moves = eval(moves)
        return moves
    except:
        print(f"Fail to extract moves: {moves}")
        return False

def extract_file_cmd(response, model):
    prompt = f"""You will be given a model-generated response regarding a file operation command in Linux system.

Your task:
Identify and extract only the actual command from the model response, removing any irrelevant or descriptive text.

Model Response:
{response}

Extracted Command:
"""
    cmds = model.generate(prompt, max_tokens=512)
    cmds = cmds.strip('`').replace('bash\n', '').split(' & ')
    return cmds

def extract_move_card(response, model):
    prompt = f"""You will be given a model-generated response describing a sequence of operations performed to cards. Each operation either adds or removes a card from pile0 or pile1.

Your task:
- Extract all valid operations and return them as a list of strings.

- Each operation must involve either adding or removing a card to or from pile0 or pile1.

- If no valid operations are found, return an empty list ([]).

Format your answer like this:
['add 6 of Hearts to pile0', 'remove King of Clubs from pile0']

Model Response:
{response}

Extracted Operations (directly return the list):
"""
    moves = model.generate(prompt, max_tokens=512)
    try:
        moves = eval(moves)
        return moves
    except:
        print(f"Fail to extract moves: {moves}")
        return False

def extract_move_chip(response, model):
    prompt = f"""You will be given a model-generated response describing a sequence of operations involving chips and cups. Each operation either adds or removes a chip from cup0 or cup1.

Your task:
- Extract all valid operations and return them as a list of strings.

- Each operation must involve either adding or removing a chip to or from cup0 or cup1.

- If no valid operations are found, return an empty list ([]).


Format your answer like this:
['add 20 to cup0', 'remove 50 cup0']

Model Response:
{response}

Extracted Operations (directly return the list):
"""
    moves = model.generate(prompt, max_tokens=512)
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

def eval_op_hrd(response, question, src_board, judge_model):
    moves = extract_move_hrd(response, judge_model)
    if not moves:
        return False, f"Invalid Operations: {moves}"
    tgt_board = extract_tgt_board_hrd(question)
    next_board = copy.deepcopy(src_board)
    for move in moves:
        next_board, is_valid = get_next_board_hrd(move, next_board)
        if not is_valid:
            continue
    return np.array_equal(np.array(next_board), tgt_board), str(moves)

def eval_op_grid(response, question, src_board, coord, judge_model):

    movemap = {'left': 'x-1', 'right': 'x+1', 'up': 'y-1', 'down': 'y+1'}

    moves = extract_move_hrd(response, judge_model)
    if not moves:
        return False, f"Invalid Operations: {moves}"
    if not all([m in movemap for m in moves]):
        return False, f"Invalid Operations: {moves}"
    moves = [movemap[m] for m in moves]

    tgt_board = extract_tgt_board_grid(question)
    boards = [copy.deepcopy(src_board)]
    for move in moves:
        next_board, coord = get_next_board_grid(move, boards[-1], coord)
        if not next_board:
            continue
        boards.append(next_board)
    return np.array_equal(np.array(boards[-1]), tgt_board), str(moves)

def eval_op_cup(response, question, src_board, judge_model):
    moves = extract_move_cup(response, judge_model)
    if not moves:
        return False, f"Invalid Operations: {moves}"
    tgt_board = extract_tgt_board_cup(question)
    next_board = copy.deepcopy(src_board)
    for move in moves:
        next_board = get_next_board_cup(move, next_board)
    return np.array_equal(np.array(next_board), tgt_board), str(moves)

def eval_op_file(response, question, states, judge_model):
    cmds = extract_file_cmd(response, judge_model)
    if (not cmds) or len(cmds)>2:                                               # at most two commands can be used
        return False, f"Invalid Operations: {response}"
    cmd_files = {'rm': set(), 'touch': set()}
    for cmd in cmds:
        if (not cmd.startswith('rm -rf')) and (not cmd.startswith('touch')):    # only `rm -rf` and `touch` can be used
            return False, f"Invalid Operations: {response}"
        match = re.search(r'\{([^}]+)\}', cmd)
        if not match:
            return False, f"Invalid Operations: {response}"
        files = set(match.group(1).split(','))
        if cmd.startswith('rm -rf'):
            cmd_files['rm'] = files
        elif cmd.startswith('touch'):
            cmd_files['touch'] = files

    if cmd_files['rm'] & cmd_files['touch']:                                    # `touch` files and `rm -rf` files should not overlap
        print("Overlap")
        return False, f"Invalid Operations: {response}"

    tgt_path, tgt_files = extract_tgt_files(question)
    pred_files = get_next_files(cmds, states, tgt_path)
    return pred_files == tgt_files, str(cmds)


def eval_op_card(response, question, src_states, model_path):
    card_types = ['Hearts', 'Clubs', 'Diamonds', 'Spades']
    card_ids = ['Ace'] + [str(i) for i in range(2, 11)] + ['Jack', 'Queen', 'King']
    all_cards = set([f"{cid} of {c_type}" for cid in card_ids for c_type in card_types])
    moves = extract_move_card(response, model_path)
    if not moves:
        return False, f"Invalid Operations: {response}"

    tgt_pile, tgt_cards = extract_tgt_cards(question)
    states = copy.deepcopy(src_states)
    for move in moves:
        if not move.startswith('add') and not move.startswith('remove'):
            return False, f"Invalid Operations: {response}"
        if not move.endswith('pile0') and not move.endswith('pile1'):
            return False, f"Invalid Operations: {response}"
        card = ' '.join(move.split()[1:-2])
        if card not in all_cards:
            return False, f"Invalid Operations: {response}"
        action = move.split()[0]
        pile = move.split()[-1]
        if pile not in states:
            return False, f"Invalid Operations: {response}"
        if action == 'add':
            states[pile].append(card)
        elif action == 'remove':
            if (not states[pile]) or (card != states[pile][0]):
                return False, f"Invalid Operations: {response}"
            states[pile].remove(card)
        
    return states[tgt_pile] == tgt_cards, str(moves)


def eval_op_chip(response, question, src_states, model_path):
    chip_types = [5, 10, 20, 50, 100]
    moves = extract_move_chip(response, model_path)
    if not moves:
        return False, f"Invalid Operations: {response}"

    tgt_cup, tgt_chips = extract_tgt_chips(question)
    states = copy.deepcopy(src_states)
    for move in moves:
        if not move.startswith('add') and not move.startswith('remove'):
            return False, f"Invalid Operations: {response}"
        if not move.endswith('cup0') and not move.endswith('cup1'):
            return False, f"Invalid Operations: {response}"
        chip = move.split()[1]
        if not chip.isdigit():
            return False, f"Invalid Operations: {response}"
        chip = int(chip)
        if chip not in chip_types:
            return False, f"Invalid Operations: {response}"
        action = move.split()[0]
        cup = move.split()[-1]
        if cup not in states:
            return False, f"Invalid Operations: {response}"
        if action == 'add':
            states[cup].append(chip)
        elif action == 'remove':
            if chip not in states[cup]:
                return False, f"Invalid Operations: {response}"
            states[cup].remove(chip)
        
    return states[tgt_cup] == tgt_chips, str(moves)

def evaluate_operation(response, data, judge_model):
    """
        Evaluate whether the operations can lead to the target state (no ground-truth answer)
    """
    data['states'] = eval(data['states'])
    if isinstance(data['coords'], str):
        data['coords'] = eval(data['coords'])
    if not response:
        return False
    if data['demo'] == "hrd":
        return eval_op_hrd(response, data['question'], \
                            data['states'][0 if data['visible_time']=='end' else -1], judge_model)
    elif data['demo'] == "grid":
        return eval_op_grid(response, data['question'], \
                            data['states'][0 if data['visible_time']=='end' else -1], 
                            data['coords'][0 if data['visible_time']=='end' else -1],
                            judge_model)
    elif data['demo'] == "cup":
        return eval_op_cup(response, data['question'], \
                            data['states'][0 if data['visible_time']=='end' else -1], judge_model)
    elif data['demo'] == "file_sys":
        states = {key: [data['states'][key][0 if data['visible_time']=="end" else -1]] for key in [f"path{i}" for i in range(int(data['num_state']))]}
        return eval_op_file(response, data['question'], \
                            states, judge_model)
    elif data['demo'] == "card":
        states = {key: data['states'][key][0 if data['visible_time']=="end" else -1] for key in [f"pile{i}" for i in range(int(data['num_state']))]}
        return eval_op_card(response, data['question'], \
                            states, judge_model)
    elif data['demo'] == "chip":
        states = {key: data['states'][key][0 if data['visible_time']=="end" else -1] for key in [f"cup{i}" for i in range(int(data['num_state']))]}
        return eval_op_chip(response, data['question'], \
                            states, judge_model)


def evaluate_video_reasoning_bench(model, line):
    assert model is not None, "Judge Model is None, VideoReasoningBench requires an LLM judge model"
    
    eval_result = {
        "question": line['question'],
        "answer": line['answer'],
        "prediction": line['prediction'],
        "demo": line['demo'],
        "dim": line['dim'],
        "visible_time": line['visible_time'],
        "num_state": line['num_state'],
        "num_operation": line['num_operation'],
        "final_ans_match": True
    }

    response = line['prediction']
    for think_tok in think_tokens:
        response = response.split(think_tok)[1] if think_tok in response else response
    response = extract_final_answer(response)
    if not response:
        eval_result['final_ans_match'] = False
        eval_result['judge_reason'] = None
        eval_result['rating'] = 0
    else:
        if line['dim']!='prediction_operation':
            judge_prompt = get_judge_prompt(line['question'], response, line['answer'])
            judge_response = model.generate(judge_prompt)
            eval_result['judge_reason'] = judge_response
            eval_result['rating'] = int(judge_response == "Correct")
        else:
            rating, eval_result['judge_reason'] = evaluate_operation(response, line, model)
            eval_result['rating'] = int(rating)
    return eval_result


def get_dimension_rating(score_file):
    data = load(score_file)
    result_dict = {'avg': [0, 0], 'final_ans_match': [0, 0]}
    for idx, item in data.iterrows():
        # dim_key = item['dim'] + '. ' + item['demo']
        dim_key = item['dim']
        demo_key = item['demo']
        if dim_key not in result_dict:
            result_dict[dim_key] = [0,0]
        if demo_key not in result_dict:
            result_dict[demo_key] = [0,0]
        result_dict[dim_key][0] += int(item['score'])
        result_dict[dim_key][1] += 1
        result_dict[demo_key][0] += int(item['score'])
        result_dict[demo_key][1] += 1
        result_dict['avg'][0] += int(item['score'])
        result_dict['avg'][1] += 1
        result_dict['final_ans_match'][0] += int(item['final_ans_match'])
        result_dict['final_ans_match'][1] += 1
    for dict_key in result_dict:
        result_dict[dict_key].append(round(100*result_dict[dict_key][0] / result_dict[dict_key][1], 1))
    save_json(result_dict, score_file.replace('.xlsx', '.json'))
    return result_dict
