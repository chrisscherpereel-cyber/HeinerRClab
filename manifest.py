# manifest.py — parameter schema for Capacity Crush (app_key "toc").
# The Juicetification Director reads this (via ?manifest=1) to build the instructor
# config UI. Nothing here is imported by the Director itself; it is served on request.
#
# Key reconciliation note: the Director/manifest uses compact, app-agnostic parameter
# names (capacities, sides, wip_cap, scrap_pct, ...). The simulation stores its inputs
# under per-station session keys (capacity_i, sides_i, wip_cap_i, scrap_pct_i) plus the
# matching scalars. juicetification.py translates the four aggregate keys into per-station
# keys on load (_director_to_snapshot); every other key here matches a snapshot key 1:1.

APP_KEY = "toc"
NAME = "Capacity Crush"
SCHEMA_VERSION = 1

MANIFEST = {
    "app_key": APP_KEY, "name": NAME, "schema_version": SCHEMA_VERSION,
    "params": {
        # per-station arrays (length N_OPS = 9)
        "capacities":  {"type": "list", "default": [1, 1, 1, 1, 1, 1, 0, 0, 0], "group": "Line", "label": "Dice per station"},
        "sides":       {"type": "list", "default": [6, 6, 6, 6, 6, 6, 0, 0, 0], "group": "Line", "label": "Faces per station"},
        "starting_inventory": {"type": "int", "default": 0, "min": 0, "group": "Line", "label": "Starting inventory"},
        "simulation_years":   {"type": "int", "default": 1, "min": 1, "max": 5, "group": "Line", "label": "Years to simulate"},
        "wip_limit_on":       {"type": "bool", "default": False, "group": "Line", "label": "Cap WIP per station"},
        "wip_cap":            {"type": "int", "default": 10, "min": 0, "max": 99999, "group": "Line", "label": "WIP cap when on"},
        "supply_reliability": {"type": "int", "default": 100, "min": 0, "max": 100, "group": "Variability", "label": "Supplier reliability %"},
        "demand_variable":    {"type": "bool", "default": False, "group": "Variability", "label": "Variable demand"},
        "demand_dice":        {"type": "int", "default": 1, "min": 1, "max": 10, "group": "Variability", "label": "Demand dice"},
        "demand_faces":       {"type": "int", "default": 6, "min": 1, "max": 100, "group": "Variability", "label": "Demand faces"},
        "reorder_point_on":   {"type": "bool", "default": False, "group": "Inventory", "label": "Manual reorder point"},
        "reorder_point":      {"type": "int", "default": 40, "min": 0, "group": "Inventory", "label": "Reorder point"},
        "scrap_on":           {"type": "bool", "default": False, "group": "Quality", "label": "Enable scrap"},
        "scrap_pct":          {"type": "int", "default": 0, "min": 0, "max": 100, "group": "Quality", "label": "Scrap % per station"},
        "fin_revenue_per_unit": {"type": "float", "default": 3.00, "group": "Economics", "label": "Revenue per unit"},
        "fin_alloc_pct":      {"type": "int", "default": 33, "min": 0, "max": 100, "group": "Economics", "label": "Fixed-cost allocation %"},
        "fin_wip_holding":    {"type": "float", "default": 0.04, "group": "Economics", "label": "WIP holding $/unit/day"},
        "fin_rmc":            {"type": "float", "default": 0.55, "group": "Economics", "label": "Raw material $/unit"},
        "fin_order_cost":     {"type": "float", "default": 25.00, "group": "Economics", "label": "$ per order"},
        "fin_order_size":     {"type": "int", "default": 150, "min": 1, "group": "Economics", "label": "Order size"},
        "fin_raw_holding":    {"type": "float", "default": 0.04, "group": "Economics", "label": "Raw holding $/unit/day"},
    }
}
