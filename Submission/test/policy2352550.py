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
        best_action = max(
            valid_actions, key=lambda action: action["score"], default=None
        )

        # Nếu không có hành động hợp lệ, chọn ngẫu nhiên
        if best_action is None:
            return self._fallback_action(products, stocks)

        return {
            "stock_idx": best_action["stock_idx"],
            "size": best_action["size"],
            "position": best_action["position"],
        }

    def _generate_valid_actions(self, products, stocks):
        """
        Sinh tất cả các hành động hợp lệ từ danh sách sản phẩm và khổ giấy.
        """
        valid_actions = []

        for product in products:
            if product["quantity"] <= 0:
                continue

            product_width, product_height = product["size"]

            for stock_index, stock in enumerate(stocks):
                stock_width, stock_height = self._get_stock_size(stock)

                if stock_width < product_width or stock_height < product_height:
                    continue

                # Tìm tất cả các vị trí đặt hợp lệ
                for x in range(stock_width - product_width + 1):
                    for y in range(stock_height - product_height + 1):
                        if self._can_place(stock, (x, y), product["size"]):
                            score = self._evaluate_position(
                                x,
                                y,
                                product_width,
                                product_height,
                                stock_width,
                                stock_height,
                            )
                            valid_actions.append(
                                {
                                    "stock_idx": stock_index,
                                    "size": product["size"],
                                    "position": (x, y),
                                    "score": score,
                                }
                            )

        return valid_actions

    def _evaluate_position(
        self, x, y, product_width, product_height, stock_width, stock_height
    ):
        """
        Đánh giá một vị trí đặt sản phẩm dựa trên nhiều tiêu chí.
        """
        score = 0

        # Ưu tiên các góc của khổ giấy
        if (x == 0 or x + product_width == stock_width) and (
            y == 0 or y + product_height == stock_height
        ):
            score += 10

        # Ưu tiên đặt gần mép
        edge_bonus = (stock_width - product_width - x) + (
            stock_height - product_height - y
        )
        score += edge_bonus * 0.5

        # Giảm điểm nếu nằm gần trung tâm
        center_x = abs((x + product_width / 2) - stock_width / 2)
        center_y = abs((y + product_height / 2) - stock_height / 2)
        score = score - (center_x + center_y) * 0.2

        return score

    def _fallback_action(self, products, stocks):
        """
        Sinh một hành động ngẫu nhiên khi không tìm thấy hành động hợp lệ.
        """
        for product in products:
            if product["quantity"] <= 0:
                continue
            product_width, product_height = product["size"]

            for stock_index, stock in enumerate(stocks):
                stock_width, stock_height = self._get_stock_size(stock)

                if stock_width < product_width or stock_height < product_height:
                    return {
                        "stock_idx": stock_index,
                        "size": product["size"],
                        "position": (0, 0),
                    }

        return {"stock_idx": 0, "size": [1, 1], "position": (0, 0)}
