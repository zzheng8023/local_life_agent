"""
Restaurant Tool — Stub implementation for the agent orchestrator.

Full restaurant search logic remains in the existing LangGraph workflow
(application/workflow.py with AmapRestaurantTool). This stub provides
a minimal interface for the orchestrator to demonstrate the restaurant
intent flow in demos.
"""

from __future__ import annotations

from typing import Any


# ================================================================
# MCP-compatible Tool Schema
# ================================================================

SEARCH_RESTAURANT_SCHEMA: dict = {
    "name": "search_restaurant",
    "description": (
        "Search for restaurants matching user preferences. "
        "Returns candidate restaurants with name, cuisine, rating, and price."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "cuisine": {
                "type": "string",
                "description": "Cuisine type, e.g. '川菜', '日料', '火锅'",
            },
            "city": {
                "type": "string",
                "description": "Search city, e.g. '北京', '上海'",
            },
            "date": {
                "type": "string",
                "description": "Target date in YYYY-MM-DD format (pre-resolved). Optional.",
            },
        },
        "required": ["cuisine", "city"],
    },
}


class RestaurantTool:
    """Stub restaurant search tool for the agent orchestrator.

    Returns mock restaurant data. In production, this would delegate
    to AmapRestaurantTool or an equivalent.
    """

    def get_name(self) -> str:
        return "search_restaurant"

    def get_description(self) -> str:
        return "Search for restaurants by cuisine, city, and optional date."

    def execute(self, **kwargs) -> dict[str, Any]:
        """ITool-compatible execute method."""
        return self.search(**kwargs)

    def search(
        self,
        cuisine: str = "",
        city: str = "上海",
        date: str = "",
    ) -> dict[str, Any]:
        """Search restaurants (mock implementation).

        Args:
            cuisine: Cuisine type.
            city: Search city.
            date: Optional date in YYYY-MM-DD format.

        Returns:
            Dict with intent info and placeholder results.
        """
        restaurants = [
            {
                "name": f"{city}{cuisine or '特色'}馆(旗舰店)",
                "cuisine": cuisine or "综合",
                "rating": 4.5,
                "avg_price": "人均120元",
            },
            {
                "name": f"老{cuisine or '字号'}酒楼({city}店)",
                "cuisine": cuisine or "综合",
                "rating": 4.2,
                "avg_price": "人均85元",
            },
        ]

        return {
            "intent": "restaurant_recommendation",
            "cuisine": cuisine,
            "city": city,
            "date": date,
            "restaurants": restaurants,
        }
