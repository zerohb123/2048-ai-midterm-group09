from __future__ import annotations

import numpy as np

from game2048.observation import Observation
from game2048.rules import move_board
from game2048.types import Move


# Ma trận trọng số góc (Snake Matrix)
WEIGHT_MATRIX = np.array([
    [4096, 1024, 256, 64],
    [16,   32,   64,  128],
    [8,    4,    2,   1],
    [0.5,  0.2,  0.1, 0.0]
], dtype=np.float64)


class Agent:
    name = "AI_2048_Midtern_Group09"

    def reset(self, seed: int | None = None) -> None:
        pass

    @staticmethod
    def evaluate(board: np.ndarray) -> float:
        # 1. Điểm ô trống
        empty_count = int(np.count_nonzero(board == 0))
        empty_score = empty_count * 280.0

        # 2. Điểm trọng số vị trí
        position_score = float(np.sum(board * WEIGHT_MATRIX))

        # 3. Điểm khả năng gộp
        horizontal_matches = (board[:, :3] == board[:, 1:]) & (board[:, :3] > 0)
        vertical_matches = (board[:3, :] == board[1:, :]) & (board[:3, :] > 0)
        merge_score = float(np.sum(board[:, :3][horizontal_matches]) + np.sum(board[:3, :][vertical_matches])) * 1.5

        return empty_score + position_score + merge_score

    def expectimax(self, board: np.ndarray, depth: int, is_max_turn: bool) -> float:
        if depth == 0:
            return self.evaluate(board)

        # ------------------- TẦNG MAX (NGƯỜI CHƠI) -------------------
        if is_max_turn:
            best_val = float("-inf")
            has_move = False

            for move in Move:
                next_board, _reward, changed = move_board(board, move)
                if changed:
                    has_move = True
                    val = self.expectimax(next_board, depth - 1, False)
                    if val > best_val:
                        best_val = val

            if not has_move:
                return self.evaluate(board)
            return best_val

        # ------------------- TẦNG CHANCE (MÁY SINH Ô) -------------------
        else:
            zero_indices = np.argwhere(board == 0)
            num_empty = len(zero_indices)
            if num_empty == 0:
                return self.evaluate(board)

            # Lấy mẫu tối đa 3 ô trống
            sample_count = min(3, num_empty)
            chosen_indices = zero_indices[:sample_count]

            expected_value = 0.0
            prob_per_cell = 1.0 / sample_count

            for r, c in chosen_indices:
                # 90% ô 2
                board[r, c] = 2
                val_2 = self.expectimax(board, depth - 1, True)
                expected_value += 0.9 * val_2 * prob_per_cell

                # 10% ô 4
                board[r, c] = 4
                val_4 = self.expectimax(board, depth - 1, True)
                expected_value += 0.1 * val_4 * prob_per_cell

                # Quay lui tại chỗ
                board[r, c] = 0

            return expected_value

    def choose_move(self, obs: Observation, time_limit_ms: float) -> Move:
        if not obs.legal_moves:
            raise ValueError("Không còn nước đi hợp lệ")

        empty_count = int(np.count_nonzero(obs.board == 0))

        # Phân bổ độ sâu động (Dynamic Depth)
        is_emergency = (obs.time_remaining_ms is not None) and (obs.time_remaining_ms < 3000.0)

        if is_emergency:
            search_depth = 1
        elif empty_count >= 6:
            search_depth = 2
        else:
            search_depth = 3

        best_move = obs.legal_moves[0]
        best_val = float("-inf")

        for move in obs.legal_moves:
            next_board, _reward, _changed = move_board(obs.board, move)
            val = self.expectimax(next_board, depth=search_depth, is_max_turn=False)
            if val > best_val:
                best_val = val
                best_move = move

        return best_move