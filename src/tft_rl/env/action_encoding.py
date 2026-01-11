# Goal: decode the simulator's 2D action-mask indices (55x38) into semantic actions [type, x1, x2].
# Why: the env internally expects semantic actions, but exposes legality as a 2D mask.

from __future__ import annotations

def grid_to_semantic_action(row: int, col: int) -> list[int]:
    """
    Mapping derived from action handler mask sizes:
      rows 0..54, cols 0..37

    Row groups (total 55):
      0: pass
      1: level
      2: refresh
      3..7: buy (shop index = row-3)
      8..44: move/sell (origin index = row-8, 37 origins: board 0..27 + bench 28..36)
      45..54: item (item index = row-45, 10 items)

    Col semantics (size 38):
      - For move/sell: col 0..36 are destination slots; col 37 is "sell".
      - For buy/level/refresh/pass: we use col=0 (col ignored otherwise).
      - For item: col 0..36 are champion targets; col 37 is unused/invalid in most cases.
    """
    if row == 0:
        return [0, 0, 0]  # pass
    if row == 1:
        return [1, 0, 0]  # level
    if row == 2:
        return [2, 0, 0]  # refresh

    if 3 <= row <= 7:
        shop_idx = row - 3
        return [3, shop_idx, 0]  # buy

    if 8 <= row <= 44:
        origin_idx = row - 8
        # col 37 corresponds to sell slot (see create_move_and_sell_action_mask: last entry is sell)
        if col == 37:
            return [4, origin_idx, 0]  # sell
        return [5, origin_idx, col]   # move

    if 45 <= row <= 54:
        item_idx = row - 45
        return [6, item_idx, col]     # item move

    raise ValueError(f"Row out of range: row={row}, col={col}")
