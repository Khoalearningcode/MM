from policy import Policy
import numpy as np


class SmartGreedy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        """Implements a smarter greedy approach"""
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        best_action = None
        best_score = float("-inf")

        # For each product that has remaining quantity
        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue

            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            # Try each stock
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Skip if product doesn't fit
                if stock_w < prod_w or stock_h < prod_h:
                    continue

                # Try each possible position
                for pos_x in range(stock_w - prod_w + 1):
                    for pos_y in range(stock_h - prod_h + 1):
                        if not self._can_place_(stock, (pos_x, pos_y), prod_size):
                            continue

                        # Calculate score based on position
                        score = self._calculate_placement_score(
                            pos_x, pos_y, prod_w, prod_h, stock_w, stock_h
                        )

                        # Update best action if this is better
                        if score > best_score:
                            best_score = score
                            best_action = {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (pos_x, pos_y),
                            }

        # If no valid action found, fall back to random valid action
        if best_action is None:
            return self._get_random_valid_action(observation)

        return best_action

    def _calculate_placement_score(
        self, pos_x, pos_y, prod_w, prod_h, stock_w, stock_h
    ):
        """Calculate a score for placing a product at a given position"""
        score = 0

        # 1. Prefer corners
        if (pos_x == 0 or pos_x + prod_w == stock_w) and (
            pos_y == 0 or pos_y + prod_h == stock_h
        ):
            score += 10  # Tăng trọng số cho các góc

        # 2. Prefer edges
        if (
            pos_x == 0
            or pos_x + prod_w == stock_w
            or pos_y == 0
            or pos_y + prod_h == stock_h
        ):
            score += 5

        # 3. Penalize central placements
        center_x = abs((pos_x + prod_w / 2) - (stock_w / 2))
        center_y = abs((pos_y + prod_h / 2) - (stock_h / 2))
        score -= (center_x + center_y) * 0.05  # Giảm phạt để tránh mất cân bằng

        # 4. Additional logic: Prefer minimizing unused area
        unused_area = (stock_w * stock_h) - (prod_w * prod_h)
        score -= unused_area * 0.01

        return score

    def _get_random_valid_action(self, observation):
        """Fallback method to get a random valid action"""
        list_prods = observation["products"]

        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue

            prod_size = prod["size"]

            # Try each stock randomly
            stock_indices = list(range(len(observation["stocks"])))
            np.random.shuffle(stock_indices)

            for stock_idx in stock_indices:
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w < prod_size[0] or stock_h < prod_size[1]:
                    continue

                # Try random positions
                for _ in range(10):  # Limit attempts
                    pos_x = np.random.randint(0, stock_w - prod_size[0] + 1)
                    pos_y = np.random.randint(0, stock_h - prod_size[1] + 1)

                    if self._can_place_(stock, (pos_x, pos_y), prod_size):
                        return {
                            "stock_idx": stock_idx,
                            "size": prod_size,
                            "position": (pos_x, pos_y),
                        }

        # If still no valid action found, return a default action
        return {"stock_idx": 0, "size": [1, 1], "position": (0, 0)}
