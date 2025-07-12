UNIVERSAL_LABEL_LIST = [
    "kettle", "microwave", "fridge", "freezer", "fridge_freezer", "dishwasher", "washing_machine",
    "tumble_dryer", "washer_dryer", "electric_heater", "toaster", "television", "desktop_computer",
    "laptop", "monitor", "router", "modem", "hi_fi", "games_console", "bread_maker", "food_mixer",
    "dehumidifier", "vivarium", "pond_pump", "space_heater", "air_conditioner", "blender", "slow_cooker",
    "water_heater", "refrigerated_drawer", "printer", "scanner", "smart_speaker", "smart_plug",
    "network_device", "projector", "charger", "coffee_machine", "iron", "hair_dryer", "lamp", "boiler",
    "oven", "fan", "radio", "amp", "vacuum_cleaner", "speakers", "htpc", "subwoofer", "set_top_box",
    "hair_straightener", "soldering_iron", "treadmill", "rice_cooker", "lighting_circuit",
    "baby_monitor", "aggregate", "other"
]


LABEL_KEYWORDS_MAP = {
    "aggregate": ["aggregate", "aggregate VA"],

    "fridge": ["fridge"],
    "freezer": ["freezer", "chest freezer"],
    "fridge_freezer": ["fridge-freezer", "fridge freezer", "fridgefreezer", "fridge_freezer"],

    "washing_machine": ["washing machine", "washing_machine"],
    "washer_dryer": ["washer dryer", "washer/dryer", "washer_dryer"],
    "dishwasher": ["dishwasher", "dish_washer"],
    "tumble_dryer": ["tumble dryer"],
    "electric_heater": ["electric heater", "heater", "electric_heater", "space_heater"],

    "kettle": ["kettle"],
    "microwave": ["microwave"],
    "toaster": ["toaster"],

    "television": ["television", "tv site", "tv", "primary_tv", "tv_dvd_digibox_lamp", "livingroom_lamp_tv"],
    "desktop_computer": ["desktop", "desktop computer", "computer site", "mjy computer", "pgm computer", "computer", "i7_desktop", "office_pc", "core2_server", "atom_pc", "data_logger_pc"],
    "laptop": ["laptop", "macbook", "laptop2"],
    "monitor": ["monitor", "lcd_office", "24_inch_lcd", "24_inch_lcd_bedroom"],

    "router": ["router", "network site", "adsl_router"],
    "modem": ["modem"],
    "hi_fi": ["hi-fi", "hi fi", "hifi", "hifi_office", "home_theatre_amp"],

    "games_console": ["games console", "game console", "xbox", "playstation", "ps4", "PS4"],
    "bread_maker": ["bread-maker", "bread maker", "breadmaker"],
    "food_mixer": ["food mixer", "k mix", "magimix", "kitchen_phone&stereo"],
    "blender": ["blender"],
    "slow_cooker": ["slow cooker"],
    "water_heater": ["water heater"],
    "dehumidifier": ["dehumidifier"],
    "refrigerated_drawer": ["refrigerated drawer"],

    "printer": ["printer", "LED_printer"],
    "scanner": ["scanner"],
    "smart_speaker": ["smart speaker"],
    "smart_plug": ["smart plug"],
    "network_device": ["network site", "computer site", "server", "server_hdd", "nas", "network_attached_storage", "gigE_&_USBhub"],

    "bread_maker": ["bread maker", "breadmaker"],
    "coffee_machine": ["coffee_machine", "nespresso_pixie"],
    "iron": ["iron", "steam_iron"],
    "hair_dryer": ["hairdryer", "hair_dryer"],
    "amp": ["amp", "amp_livingroom"],
    "projector": ["projector"],

    "vacuum_cleaner": ["vacuum_cleaner", "hoover"],
    "charger": ["charger", "ipad_charger", "samsung_charger", "bedroom_chargers", "battery_charger"],

    "lamp": [
        "lamp", "livingroom_s_lamp", "livingroom_s_lamp2", "kitchen_dt_lamp", "bedroom_ds_lamp",
        "bedroom_d_lamp", "office_lamp1", "office_lamp2", "office_lamp3", "childs_table_lamp", "childs_ds_lamp",
        "utilityrm_lamp", "kitchen_lamp2"
    ],

    "boiler": ["boiler", "gas_boiler"],
    "oven": ["oven", "gas_oven"],
    "radio": ["dab_radio", "kitchen_radio", "tv_dvd_digibox_lamp", "kettle_radio"],
    "fan": ["office_fan"],
    "amp": ["amp_livingroom"],

    "speakers": ["speakers", "stereo_speakers_bedroom"],
    "htpc": ["htpc"],
    "subwoofer": ["subwoofer", "subwoofer_livingroom"],
    "set_top_box": ["sky_hd_box", "set top box"],
    "hair_straightener": ["straighteners", "hair_straightener"],
    "soldering_iron": ["soldering_iron"],
    "treadmill": ["running_machine", "treadmill"],
    "rice_cooker": ["rice_cooker"],
    "lighting_circuit": ["lighting_circuit"],
    "baby_monitor": ["baby_monitor_tx", "baby monitor"]
}
