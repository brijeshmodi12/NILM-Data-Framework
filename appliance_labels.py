UNIVERSAL_LABEL_LIST = [
    "kettle", "microwave", "fridge", "freezer", "fridge_freezer", "dishwasher", "washing_machine", "tumble_dryer",
    "washer_dryer", "electric_heater", "toaster", "television", "desktop_computer", "laptop", "monitor", "router",
    "modem", "hi_fi", "games_console", "bread_maker", "food_mixer", "dehumidifier", "vivarium", "pond_pump",
    "space_heater", "air_conditioner", "blender", "slow_cooker", "water_heater", "refrigerated_drawer",
    "printer", "scanner", "smart_speaker", "smart_plug", "network_device", "aggregate", "other"
]

LABEL_KEYWORDS_MAP = {
    # universal_label : [ dataset keywords ]
    "aggregate":["aggregate"],
    "fridge": ["fridge"],
    "freezer": ["freezer", "chest freezer"],
    "fridge_freezer": ["fridge-freezer", "fridge freezer", "fridgefreezer"],
    "washer_dryer": ["washer dryer", "washer/dryer"],
    "washing_machine": ["washing machine"],
    "dishwasher": ["dishwasher"],
    "tumble_dryer": ["tumble dryer"],
    "electric_heater": ["electric heater", "heater"],
    "kettle": ["kettle"],
    "microwave": ["microwave"],
    "toaster": ["toaster"],
    "television": ["television", "tv site", "tv"],
    "desktop_computer": ["desktop", "desktop computer", "computer site", "mjy computer", "pgm computer", "computer"],
    "laptop": ["laptop", "macbook"],
    "router": ["router", "network site"],
    "modem": ["modem"],
    "monitor": ["monitor"],
    "hi_fi": ["hi-fi", "hi fi", "hifi"],
    "games_console": ["games console", "game console", "xbox", "playstation"],
    "bread_maker": ["bread-maker", "bread maker"],
    "food_mixer": ["food mixer", "k mix", "magimix"],
    "blender": ["blender"],
    "dehumidifier": ["dehumidifier"],
    "vivarium": ["vivarium"],
    "pond_pump": ["pond pump"],
    "network_device": ["network site", "computer site"],
}
