# app.py

import streamlit as st
from tic_tac_toe import TicTacToe
from q_agent import QLearningAgent
from q_agent_advanced import QAgentAdvanced
from minimax_agent import best_move, check_winner
from learning_lab import run_training_experiment
import random

st.set_page_config(
    page_title="Tic-Tac-Toe AI Lab",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
    body {
        background: #0b0f14;
        color: #e8e8e8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f2f2f2;
        letter-spacing: 0.3px;
    }
    h2, h3 {
        color: #e0e0e0;
    }
    .stButton>button {
        background: #151b24;
        color: #e8e8e8;
        border: 1px solid #2a3340;
        border-radius: 10px;
        padding: 0.4rem 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #111821;
        color: #d8d8d8;
        border: 1px solid #1f2936;
        border-radius: 8px;
        padding: 0.4rem 0.9rem;
    }
    .info-box {
        background: #111821;
        border: 1px solid #1f2936;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.7rem 0 1rem 0;
        color: #cfd6dd;
        font-size: 0.93rem;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Tic-Tac-Toe AI Lab")
st.markdown(
    '<div class="info-box">'
    'Explore how <b>classical search (Minimax)</b> and <b>reinforcement learning (Q-learning)</b> behave in a solved game. '
    'Train agents, view learning curves, and inspect move-by-move reasoning.'
    '</div>',
    unsafe_allow_html=True
)

# Load agents
@st.cache_resource
def load_q_basic():
    agent = QLearningAgent()
    agent.load("q_table.json")
    return agent

@st.cache_resource
def load_q_advanced():
    agent = QAgentAdvanced()
    agent.load("q_table_advanced.json")
    return agent

q_basic = load_q_basic()
q_advanced = load_q_advanced()

# Session state for Play tab
if "board" not in st.session_state:
    st.session_state.board = [0] * 9
    st.session_state.done = False
    st.session_state.winner = None
    st.session_state.human_symbol = 1
    st.session_state.ai_symbol = 2
    st.session_state.turn = 1
    st.session_state.play_mode = "human_vs_minimax"
    st.session_state.stats = {
        "human_vs_minimax": {"wins": 0, "draws": 0, "losses": 0},
        "human_vs_q_advanced": {"wins": 0, "draws": 0, "losses": 0},
    }
    st.session_state.move_history = []
    st.session_state.last_ai_scores = None

def reset_game():
    st.session_state.board = [0] * 9
    st.session_state.done = False
    st.session_state.winner = None
    st.session_state.turn = 1
    st.session_state.move_history = []
    st.session_state.last_ai_scores = None

def make_move(action, player):
    env = TicTacToe()
    env.board = st.session_state.board.copy()
    env.done = st.session_state.done
    env.winner = st.session_state.winner
    next_state, _, done = env.step(action, player)
    st.session_state.board = list(next_state)
    st.session_state.done = env.done
    st.session_state.winner = env.winner
    sym = {0: ".", 1: "X", 2: "O"}
    st.session_state.move_history.append(f"{len(st.session_state.move_history)+1}. {sym[player]} → pos {action}")

# Tabs
tab_play, tab_lab, tab_reason = st.tabs(["▶️ Play", "📈 Learning Lab", "🔍 AI Reasoning"])

# ---------------- PLAY TAB ----------------
with tab_play:
    st.subheader("Play against AI")

    st.sidebar.header("⚙️ Play Settings")
    mode = st.sidebar.radio(
        "Mode",
        ["You vs Unbeatable AI (Minimax)", "You vs Strong Q-Agent (Advanced RL)"],
        index=0,
        key="playmode"
    )
    st.session_state.play_mode = "human_vs_minimax" if "Minimax" in mode else "human_vs_q_advanced"

    play_as = st.sidebar.radio("Play as", ["X", "O"], index=0, key="playas")
    st.session_state.human_symbol = 1 if play_as == "X" else 2
    st.session_state.ai_symbol = 2 if play_as == "X" else 1

    if st.sidebar.button("Reset Game", key="resetplay"):
        reset_game()
        st.rerun()

    board = st.session_state.board
    symbols = {0: " ", 1: "X", 2: "O"}

    # Highlight winning line
    winning_line = None
    if st.session_state.done and st.session_state.winner in (1,2):
        wins = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        for a,b,c in wins:
            if board[a] == board[b] == board[c] == st.session_state.winner:
                winning_line = (a,b,c)
                break

    def render_board():
        for row in range(3):
            cols = st.columns(3)
            for col in range(3):
                i = row * 3 + col
                val = board[i]
                label = symbols[val]

                disable = True
                if st.session_state.play_mode in ["human_vs_minimax", "human_vs_q_advanced"]:
                    if val == 0 and not st.session_state.done and st.session_state.turn == st.session_state.human_symbol:
                        disable = False

                btn = cols[col].button(label, key=f"cell_{i}", disabled=disable, use_container_width=True)
                if not disable and btn:
                    make_move(i, st.session_state.human_symbol)

                    if not st.session_state.done:
                        ai_moves = [j for j, v in enumerate(st.session_state.board) if v == 0]
                        if ai_moves:
                            if st.session_state.play_mode == "human_vs_minimax":
                                ai_action = best_move(
                                    st.session_state.board,
                                    st.session_state.ai_symbol,
                                    st.session_state.human_symbol
                                )
                            else:
                                ai_action = q_advanced.choose_action(
                                    st.session_state.board,
                                    ai_moves
                                )
                            make_move(ai_action, st.session_state.ai_symbol)

                    if st.session_state.done:
                        if st.session_state.winner == st.session_state.human_symbol:
                            st.session_state.stats[st.session_state.play_mode]["wins"] += 1
                        elif st.session_state.winner is None:
                            st.session_state.stats[st.session_state.play_mode]["draws"] += 1
                        else:
                            st.session_state.stats[st.session_state.play_mode]["losses"] += 1
                    st.rerun()

    render_board()

    if winning_line:
        a,b,c = winning_line
        st.success(f"Winner: {'X' if st.session_state.winner==1 else 'O'} (line: {a}-{b}-{c})")

    if st.session_state.done:
        if st.session_state.winner == st.session_state.human_symbol:
            if st.session_state.play_mode == "human_vs_minimax":
                st.success("You won! (Extremely unlikely against perfect play.)")
            else:
                st.success("You won against the advanced Q-agent!")
        elif st.session_state.winner is None:
            st.info("It's a draw!")
        else:
            st.error("AI won!")
    else:
        st.write("Your turn." if not st.session_state.done else "AI thinking…")

    st.markdown(
        '<div class="info-box">'
        '<b>Move history:</b> ' +
        (" – ".join(st.session_state.move_history) if st.session_state.move_history else "No moves yet.") +
        '</div>',
        unsafe_allow_html=True
    )

    # Store last position + AI scores for Reasoning tab
    if not st.session_state.done:
        ai_moves = [j for j, v in enumerate(board) if v == 0]
        if ai_moves:
            if st.session_state.play_mode == "human_vs_minimax":
                scores = {}
                for m in ai_moves:
                    test_board = board.copy()
                    test_board[m] = st.session_state.ai_symbol
                    # crude: use minimax value via best_move comparison
                    optimal = best_move(board.copy(), st.session_state.ai_symbol, st.session_state.human_symbol)
                    scores[m] = "optimal" if m == optimal else "suboptimal"
                st.session_state.last_ai_scores = {"type": "minimax", "scores": scores, "board": board}
            else:
                q = q_advanced._get_q_dict(board)
                st.session_state.last_ai_scores = {"type": "q", "scores": {m: round(q.get(m,0),2) for m in ai_moves}, "board": board}

# ---------------- LEARNING LAB TAB ----------------
with tab_lab:
    st.subheader("Training Experiment: Q-Agent Learning Curve")

    st.markdown(
        '<div class="info-box">'
        'Train the advanced Q-agent in chunks and evaluate its win rate over time. '
        'This shows how reinforcement learning improves with experience.'
        '</div>',
        unsafe_allow_html=True
    )

    total_episodes = st.slider("Total episodes", 5000, 50000, 10000, step=5000, key="totalepi")
    eval_every = st.slider("Evaluate every (episodes)", 500, 2000, 1000, step=500, key="evevery")
    eval_episodes = st.slider("Evaluation games", 100, 500, 200, step=100, key="evepi")
    opponent_type = st.radio("Opponent for evaluation", ["random", "minimax"], index=0, key="opptype")

    if st.button("Run Training Experiment"):
        with st.spinner("Training and evaluating… this may take a few seconds."):
            history, agent = run_training_experiment(
                total_episodes=total_episodes,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                opponent_type=opponent_type
            )
            agent.save("q_table_advanced.json")

            # Prepare chart data
            import pandas as pd
            df = pd.DataFrame(history)
            df["win_rate"] = df["wins"] / (df["wins"] + df["draws"] + df["losses"])
            df["draw_rate"] = df["draws"] / (df["wins"] + df["draws"] + df["losses"])
            df["loss_rate"] = df["losses"] / (df["wins"] + df["draws"] + df["losses"])

            st.markdown("### Win/Draw/Loss rates vs " + opponent_type)
            st.line_chart(df[["win_rate", "draw_rate", "loss_rate"]])

            st.markdown(
                '<div class="info-box">'
                'The curves show how the Q-agent’s performance changes as it trains. '
                'Against a random opponent, win rate should rise. Against Minimax, the agent learns to force more draws.'
                '</div>',
                unsafe_allow_html=True
            )

# ---------------- AI REASONING TAB ----------------
with tab_reason:
    st.subheader("AI Move Reasoning")

    st.markdown(
        '<div class="info-box">'
        'Inspect how the AI evaluates moves for the last position from the Play tab. '
        'This helps you understand the difference between search-based and learning-based decisions.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.last_ai_scores is None:
        st.info("Play a move in the Play tab to generate AI reasoning data.")
    else:
        data = st.session_state.last_ai_scores
        board = data["board"]
        symbols = {0: "·", 1: "X", 2: "O"}

        st.write("Current board:")
        # show 3x3 text board
        rows = [
            " ".join(symbols[board[i]] for i in range(3)),
            " ".join(symbols[board[i]] for i in range(3,6)),
            " ".join(symbols[board[i]] for i in range(6,9)),
        ]
        st.text("\n".join(rows))

        if data["type"] == "minimax":
            st.write("Minimax evaluation for each legal move:")
            scores = data["scores"]
            lines = [f"Move {m}: {s}" for m,s in scores.items()]
            st.text("\n".join(lines))
            st.markdown(
                '<div class="info-box">'
                '<b>Interpretation:</b> “optimal” moves lead to the best guaranteed outcome (win or draw) assuming perfect play. '
                'Minimax chooses one of these optimal moves.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.write("Q-values for each legal move (higher = better according to the Q-agent):")
            scores = data["scores"]
            lines = [f"Move {m}: {s}" for m,s in scores.items()]
            st.text("\n".join(lines))
            st.markdown(
                '<div class="info-box">'
                '<b>Interpretation:</b> The Q-agent has learned these values from self-play. '
                'Higher Q-values indicate moves that historically led to better outcomes (wins/draws).'
                '</div>',
                unsafe_allow_html=True
            )

st.markdown("---")
st.markdown(
    '<div class="info-box">'
    '<b>Why this matters:</b> Minimax is perfect but requires search. Q-learning learns from experience and generalizes, '
    'but may not reach perfect play. This lab lets you compare both approaches in a solved game.'
    '</div>',
    unsafe_allow_html=True
)