# minimax_agent.py

def check_winner(board):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in wins:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if all(v != 0 for v in board):
        return "draw"
    return None

def minimax(board, is_maximizing, ai_symbol, human_symbol):
    res = check_winner(board)
    if res == ai_symbol:
        return 10
    if res == human_symbol:
        return -10
    if res == "draw":
        return 0

    if is_maximizing:
        best = -float("inf")
        for i in range(9):
            if board[i] == 0:
                board[i] = ai_symbol
                score = minimax(board, False, ai_symbol, human_symbol)
                board[i] = 0
                best = max(best, score)
        return best
    else:
        best = float("inf")
        for i in range(9):
            if board[i] == 0:
                board[i] = human_symbol
                score = minimax(board, True, ai_symbol, human_symbol)
                board[i] = 0
                best = min(best, score)
        return best

def best_move(board, ai_symbol, human_symbol):
    best_score = -float("inf")
    move = None
    for i in range(9):
        if board[i] == 0:
            board[i] = ai_symbol
            score = minimax(board, False, ai_symbol, human_symbol)
            board[i] = 0
            if score > best_score:
                best_score = score
                move = i
    return move