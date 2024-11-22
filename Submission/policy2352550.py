from policy import Policy
import numpy as np


class SmartGreedy(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info):
        """
        Tìm kiếm hành động tốt nhất bằng cách tiếp cận tham lam.
        """
        products = observation["products"]
        stocks = observation["stocks"]

        # Tìm tất cả các hành động hợp lệ
        valid_actions = self._generate_valid_actions(products, stocks)

        # Chọn hành động tốt nhất dựa trên điểm đánh giá
        best_action = max(valid_actions, key=lambda x: x["score"], default=None)

        # Nếu không có hành động hợp lệ, chọn ngẫu nhiên
        if best_action is None:
            return self._fallback_action(products, stocks)

        return {
            "stock_idx": best_action["stock_idx"],
            "size": best_action["size"],
            "position": best_action["position"]
        }

    def _generate_valid_actions(self, products, stocks):
        """
        Sinh tất cả các hành động hợp lệ từ danh sách sản phẩm và khổ giấy.
        """
        valid_actions = []

        for prod in products:
            if prod["quantity"] <= 0:
                continue

            prod_w, prod_h = prod["size"]

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size(stock)

                if stock_w < prod_w or stock_h < prod_h:
                    continue

                # Tìm tất cả các vị trí đặt hợp lệ
                for pos_x in range(stock_w - prod_w + 1):
                    for pos_y in range(stock_h - prod_h + 1):
                        if self._can_place(stock, (pos_x, pos_y), prod["size"]):
                            score = self._evaluate_position(
                                pos_x, pos_y, prod_w, prod_h, stock_w, stock_h
                            )
                            valid_actions.append({
                                "stock_idx": stock_idx,
                                "size": prod["size"],
                                "position": (pos_x, pos_y),
                                "score": score
                            })

        return valid_actions

    def _evaluate_position(self, pos_x, pos_y, prod_w, prod_h, stock_w, stock_h):
        """
        Đánh giá một vị trí đặt sản phẩm dựa trên nhiều tiêu chí.
        """
        score = 0

        # Ưu tiên các góc của khổ giấy
        if (pos_x == 0 or pos_x + prod_w == stock_w) and \
           (pos_y == 0 or pos_y + prod_h == stock_h):
            score += 10

        # Ưu tiên đặt gần mép
        edge_bonus = (stock_w - prod_w - pos_x) + (stock_h - prod_h - pos_y)
        score += edge_bonus * 0.5

        # Giảm điểm nếu nằm gần trung tâm
        center_x = abs((pos_x + prod_w / 2) - stock_w / 2)
        center_y = abs((pos_y + prod_h / 2) - stock_h / 2)
        score -= (center_x + center_y) * 0.2

        return score

    def _fallback_action(self, products, stocks):
        """
        Sinh một hành động ngẫu nhiên khi không tìm thấy hành động hợp lệ.
        """
        for prod in products:
            if prod["quantity"] > 0:
                prod_w, prod_h = prod["size"]

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        return {
                            "stock_idx": stock_idx,
                            "size": prod["size"],
                            "position": (0, 0)
                        }

        return {
            "stock_idx": 0,
            "size": [1, 1],
            "position": (0, 0)
        }