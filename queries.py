"""
Categorized natural-language query benchmark for WoT discovery retrieval.

This module is the single source of truth for evaluation queries. It is consumed
by the centralized benchmark (eval_lib), the federated experiments, and the
deployment study, so that every experiment evaluates the *same* queries.

Each query is a dict:
    id          : stable identifier (category prefix + number)
    category    : one of CATEGORIES
    query       : the natural-language request string
    expected    : list of accessedNodeAddress endpoints that count as correct.
                  - exactly one element for unambiguous queries
                  - several elements for ambiguous queries (any one is a hit)
                  - empty list [] for no-answer queries (system should abstain)
    ambiguous   : True if several distinct devices legitimately satisfy the query
    provenance  : "llopis2025_table11" for the 10 ChatGPT-5 paraphrases reused
                  verbatim from Llopis et al. (2025) Table 11; "authored" otherwise
    note        : optional free-text rationale

Endpoints are taken from the 286 unique trace-derived records in
mainSimulationAccessTraces.csv. Helper dicts below map (service, location) to the
operation-level endpoint so query authoring stays consistent with the dataset.

Room/agent map (rooms 1..10 -> agents 20..29):
    room_1=20 room_2=21 room_3=22 room_4=23 room_5=24
    room_6=25 room_7=26 room_8=27 room_9=28 room_10=29
"""

from __future__ import annotations

CATEGORIES = [
    "templated",
    "paraphrased",
    "synonym",
    "abstract",
    "ambiguous_location",
    "ambiguous_device",
    "ambiguous_instance",
    "no_answer",
]

# ---------------------------------------------------------------------------
# Endpoint inventory (operation-level endpoints), derived from the dataset.
# ---------------------------------------------------------------------------

LIGHT = {  # lightControler -> .../lightOn
    "Bathroom": "/agent6/lightcontrol6/lightOn",
    "Bedroom": "/agent13/lightcontrol13/lightOn",
    "BedroomChildren": "/agent1/lightcontrol1/lightOn",
    "BedroomParents": "/agent2/lightcontrol2/lightOn",
    "Dinningroom": "/agent3/lightcontrol3/lightOn",
    "Entrance": "/agent12/lightcontrol12/lightOn",
    "Garage": "/agent5/lightcontrol5/lightOn",
    "Kitchen": "/agent4/lightcontrol4/lightOn",
    "Livingroom": "/agent10/lightcontrol10/lightOn",
    "Showerroom": "/agent14/lightcontrol14/lightOn",
    "Watterroom": "/agent11/lightcontrol11/lightOn",
    "room_1": "/agent20/lightcontrol20/lightOn",
    "room_2": "/agent21/lightcontrol21/lightOn",
    "room_3": "/agent22/lightcontrol22/lightOn",
    "room_4": "/agent23/lightcontrol23/lightOn",
    "room_5": "/agent24/lightcontrol24/lightOn",
    "room_6": "/agent25/lightcontrol25/lightOn",
    "room_7": "/agent26/lightcontrol26/lightOn",
    "room_8": "/agent27/lightcontrol27/lightOn",
    "room_9": "/agent28/lightcontrol28/lightOn",
    "room_10": "/agent29/lightcontrol29/lightOn",
}

MOVE = {  # movementSensor -> .../movement
    "Bathroom": "/agent6/movement6/movement",
    "Bedroom": "/agent13/movement13/movement",
    "BedroomChildren": "/agent1/movement1/movement",
    "BedroomParents": "/agent2/movement2/movement",
    "Dinningroom": "/agent3/movement3/movement",
    "Entrance": "/agent12/movement12/movement",
    "Garage": "/agent5/movement5/movement",
    "Kitchen": "/agent4/movement4/movement",
    "Livingroom": "/agent10/movement10/movement",
    "Showerroom": "/agent14/movement14/movement",
    "Watterroom": "/agent11/movement11/movement",
    "room_1": "/agent20/movement20/movement",
    "room_2": "/agent21/movement21/movement",
    "room_3": "/agent22/movement22/movement",
    "room_4": "/agent23/movement23/movement",
    "room_5": "/agent24/movement24/movement",
    "room_6": "/agent25/movement25/movement",
    "room_7": "/agent26/movement26/movement",
    "room_8": "/agent27/movement27/movement",
    "room_9": "/agent28/movement28/movement",
    "room_10": "/agent29/movement29/movement",
}

TEMP = {  # sensorService (temperature) -> bare endpoint (no sub-path)
    "Bathroom": "/agent6/tempin6",
    "Bedroom": "/agent13/tempin13",
    "BedroomChildren": "/agent1/tempin1",
    "BedroomParents": "/agent2/tempin2",
    "Dinningroom": "/agent3/tempin3",
    "Entrance": "/agent12/tempin12",
    "Garage": "/agent5/tempin5",
    "Kitchen": "/agent4/tempin4",
    "Livingroom": "/agent10/tempin10",
    "Showerroom": "/agent14/tempin14",
    "Watterroom": "/agent11/tempin11",
    "room_1": "/agent20/tempin20",
    "room_2": "/agent21/tempin21",
    "room_3": "/agent22/tempin22",
    "room_4": "/agent23/tempin23",
    "room_5": "/agent24/tempin24",
    "room_6": "/agent25/tempin25",
    "room_7": "/agent26/tempin26",
    "room_8": "/agent27/tempin27",
    "room_9": "/agent28/tempin28",
    "room_10": "/agent29/tempin29",
}

LOCK = {  # doorLockService -> .../open
    "Dinningroom": "/agent3/doorlock1/open",
    "Entrance": "/agent12/doorlock3/open",
    "Livingroom": "/agent10/doorlock2/open",
    "room_1": "/agent20/doorlock4/open",
    "room_2": "/agent21/doorlock5/open",
}

THERMO = {  # thermostat -> heatingcontrol (read endpoint .../heatingOn where present)
    "Dinningroom": "/agent3/heatingcontrol1/heatingOn",
    "Entrance": "/agent12/heatingcontrol3/heatingOn",
    "Watterroom": "/agent11/heatingcontrol2/heatingOn",
    "room_6": "/agent25/heatingcontrol4",  # only register/write exist for this one
}

BATT = {  # batteryService -> .../charge  (lists: several batteries per location)
    "Entrance": ["/agent12/battery5/charge", "/agent12/battery6/charge"],
    "Garage": ["/agent5/battery1/charge", "/agent5/battery2/charge"],
    "Kitchen": ["/agent4/battery3/charge"],
    "Watterroom": ["/agent11/battery4/charge"],
}

WASH = {  # washingService -> .../washing
    "Bathroom": "/agent6/washingmachine1/washing",
    "Showerroom": "/agent14/washingmachine3/washing",
    "Watterroom": "/agent11/washingmachine3/washing",
}

# Convenience groups for ambiguous queries
ALL_LIGHTS = list(LIGHT.values())
BEDROOM_LIGHTS = [LIGHT["Bedroom"], LIGHT["BedroomChildren"], LIGHT["BedroomParents"]]
BEDROOM_TEMPS = [TEMP["Bedroom"], TEMP["BedroomChildren"], TEMP["BedroomParents"]]
BEDROOM_MOVES = [MOVE["Bedroom"], MOVE["BedroomChildren"], MOVE["BedroomParents"]]
ALL_MOVES = list(MOVE.values())
ALL_LOCKS = list(LOCK.values())


def _q(qid, category, query, expected, ambiguous=False,
       provenance="authored", note=""):
    if isinstance(expected, str):
        expected = [expected]
    return {
        "id": qid,
        "category": category,
        "query": query,
        "expected": expected,
        "ambiguous": ambiguous,
        "provenance": provenance,
        "note": note,
    }


QUERIES = [
    # -----------------------------------------------------------------
    # 1) Templated — match the generated sentence pattern; unambiguous.
    # -----------------------------------------------------------------
    _q("tmpl01", "templated", "I need to write lightControler in Kitchen", LIGHT["Kitchen"]),
    _q("tmpl02", "templated", "I need to read movementSensor in Entrance", MOVE["Entrance"]),
    _q("tmpl03", "templated", "I need to read sensorService in BedroomParents", TEMP["BedroomParents"]),
    _q("tmpl04", "templated", "I need to write doorLockService in Entrance", LOCK["Entrance"]),
    _q("tmpl05", "templated", "I need to read washingService in Showerroom", WASH["Showerroom"]),
    _q("tmpl06", "templated", "I need to read thermostat in Dinningroom", THERMO["Dinningroom"]),
    _q("tmpl07", "templated", "I need to write lightControler in Garage", LIGHT["Garage"]),
    _q("tmpl08", "templated", "I need to read movementSensor in room_5", MOVE["room_5"]),
    _q("tmpl09", "templated", "I need to read sensorService in room_8", TEMP["room_8"]),
    _q("tmpl10", "templated", "I need to write doorLockService in Livingroom", LOCK["Livingroom"]),
    _q("tmpl11", "templated", "I need to read lightControler in Watterroom", LIGHT["Watterroom"]),

    # -----------------------------------------------------------------
    # 2) Paraphrased — the 10 ChatGPT-5 paraphrases from Llopis 2025
    #    Table 11 (reused verbatim), plus authored paraphrases.
    # -----------------------------------------------------------------
    _q("para01", "paraphrased", "Could you please turn on the bedroom lights in my parents' room?",
       LIGHT["BedroomParents"], provenance="llopis2025_table11"),
    _q("para02", "paraphrased", "Switch on the dining room lights, please.",
       LIGHT["Dinningroom"], provenance="llopis2025_table11"),
    _q("para03", "paraphrased", "Turn on the lights in the kids' bedroom.",
       LIGHT["BedroomChildren"], provenance="llopis2025_table11"),
    _q("para04", "paraphrased", "Please switch on the kitchen lights.",
       LIGHT["Kitchen"], provenance="llopis2025_table11"),
    _q("para05", "paraphrased", "Check if there's any movement in the kitchen.",
       MOVE["Kitchen"], provenance="llopis2025_table11"),
    _q("para06", "paraphrased", "What's the current temperature in my parents' bedroom?",
       TEMP["BedroomParents"], provenance="llopis2025_table11"),
    _q("para07", "paraphrased", "Is there anyone moving in the parents' bedroom right now?",
       MOVE["BedroomParents"], provenance="llopis2025_table11"),
    _q("para08", "paraphrased", "Could you tell me the kitchen temperature?",
       TEMP["Kitchen"], provenance="llopis2025_table11"),
    _q("para09", "paraphrased", "How warm is it in the children's bedroom?",
       TEMP["BedroomChildren"], provenance="llopis2025_table11"),
    _q("para10", "paraphrased", "Check for motion in the children's bedroom.",
       MOVE["BedroomChildren"], provenance="llopis2025_table11"),
    # authored paraphrases (new wording, unambiguous target)
    _q("para11", "paraphrased", "Please unlock the front entrance door.", LOCK["Entrance"]),
    _q("para12", "paraphrased", "Can you start the washing machine in the shower room?", WASH["Showerroom"]),
    _q("para13", "paraphrased", "Switch the garage light on for me.", LIGHT["Garage"]),
    _q("para14", "paraphrased", "Tell me the temperature in the living room.", TEMP["Livingroom"]),
    _q("para15", "paraphrased", "Is the heating on in the dining room?", THERMO["Dinningroom"]),
    _q("para16", "paraphrased", "Has the entrance motion sensor seen anyone?", MOVE["Entrance"]),

    # -----------------------------------------------------------------
    # 3) Synonym-heavy — synonyms for the indexed service vocabulary.
    # -----------------------------------------------------------------
    _q("syn01", "synonym", "How warm is the kitchen right now?", TEMP["Kitchen"],
       note="'warm' vs sensorService/temperature"),
    _q("syn02", "synonym", "Turn on the lamp in the living room.", LIGHT["Livingroom"],
       note="'lamp' vs lightControler"),
    _q("syn03", "synonym", "Is anyone moving about in the garage?", MOVE["Garage"],
       note="'moving about' vs movementSensor"),
    _q("syn04", "synonym", "Illuminate the dining room.", LIGHT["Dinningroom"],
       note="'illuminate' vs light"),
    _q("syn05", "synonym", "What's the charge level of the kitchen battery?", BATT["Kitchen"]),
    _q("syn06", "synonym", "Detect motion in the parents' bedroom.", MOVE["BedroomParents"]),
    _q("syn07", "synonym", "Open the entrance latch.", LOCK["Entrance"],
       note="'latch' vs doorLockService"),
    _q("syn08", "synonym", "How hot is it in the children's bedroom?", TEMP["BedroomChildren"]),
    _q("syn09", "synonym", "Run the laundry in the bathroom.", WASH["Bathroom"],
       note="'laundry' vs washingService"),
    _q("syn10", "synonym", "Crank up the heating in the entrance.", THERMO["Entrance"]),
    _q("syn11", "synonym", "Light up the shower room.", LIGHT["Showerroom"]),

    # -----------------------------------------------------------------
    # 4) Abstract / indirect — intent without naming the device type.
    # -----------------------------------------------------------------
    _q("abs01", "abstract", "It's too dark in the kitchen.", LIGHT["Kitchen"],
       note="implies turning on the light"),
    _q("abs02", "abstract", "Is anybody home? Check the entrance.", MOVE["Entrance"]),
    _q("abs03", "abstract", "It feels chilly in the dining room.", THERMO["Dinningroom"],
       note="implies turning up the heating; scored against the thermostat's operation endpoint"),
    _q("abs04", "abstract", "I can't see anything in the garage.", LIGHT["Garage"]),
    _q("abs05", "abstract", "Make sure the front door is secured.", LOCK["Entrance"]),
    _q("abs06", "abstract", "My laundry needs doing in the shower room.", WASH["Showerroom"]),
    _q("abs07", "abstract", "Is it stuffy and hot in the living room?", TEMP["Livingroom"],
       note="asks for the room's temperature reading (no thermostat exists in Livingroom)"),
    _q("abs08", "abstract", "Did something just move in the kids' room?", MOVE["BedroomChildren"]),
    _q("abs09", "abstract", "The phone says the kitchen device is running low.", BATT["Kitchen"]),
    _q("abs10", "abstract", "I'd like some light in the living room.", LIGHT["Livingroom"]),

    # -----------------------------------------------------------------
    # 5) Ambiguous location — location underspecified; several devices ok.
    # -----------------------------------------------------------------
    _q("ambl01", "ambiguous_location", "Turn on the bedroom light.", BEDROOM_LIGHTS, ambiguous=True,
       note="three bedrooms: Bedroom, BedroomChildren, BedroomParents"),
    _q("ambl02", "ambiguous_location", "What's the temperature in the bedroom?", BEDROOM_TEMPS, ambiguous=True),
    _q("ambl03", "ambiguous_location", "Is there movement in the bedroom?", BEDROOM_MOVES, ambiguous=True),
    _q("ambl04", "ambiguous_location", "Switch on the upstairs light.",
       [LIGHT[f"room_{i}"] for i in range(1, 11)], ambiguous=True,
       note="'upstairs' loosely maps to the numbered rooms"),
    _q("ambl05", "ambiguous_location", "Read a battery charge.",
       [e for v in BATT.values() for e in v], ambiguous=True),
    _q("ambl06", "ambiguous_location", "Unlock a door.", ALL_LOCKS, ambiguous=True),

    # -----------------------------------------------------------------
    # 6) Ambiguous device — device underspecified across many locations.
    # -----------------------------------------------------------------
    _q("ambd01", "ambiguous_device", "Turn on the light.", ALL_LIGHTS, ambiguous=True),
    _q("ambd02", "ambiguous_device", "Is there any movement anywhere?", ALL_MOVES, ambiguous=True),
    _q("ambd03", "ambiguous_device", "Switch on a light somewhere.", ALL_LIGHTS, ambiguous=True),
    _q("ambd04", "ambiguous_device", "Check a motion sensor.", ALL_MOVES, ambiguous=True),
    _q("ambd05", "ambiguous_device", "What's the temperature?", list(TEMP.values()), ambiguous=True),

    # -----------------------------------------------------------------
    # 7) Ambiguous instance — location and device type both specified,
    #    but that location has more than one physical device of that
    #    type, so any of them is an acceptable answer.
    # -----------------------------------------------------------------
    _q("ambi01", "ambiguous_instance", "I need to read batteryService in Garage", BATT["Garage"],
       ambiguous=True, note="two batteries in the garage; either is acceptable"),
    _q("ambi02", "ambiguous_instance", "Power reading for the garage cells.", BATT["Garage"],
       ambiguous=True, note="two batteries in the garage; either is acceptable"),

    # -----------------------------------------------------------------
    # 8) No-answer — device type or capability not present in the corpus.
    #    Correct behaviour is to abstain (expected == []).
    # -----------------------------------------------------------------
    _q("no01", "no_answer", "Turn on the television in the living room.", [],
       note="no TV/media device in the corpus"),
    _q("no02", "no_answer", "Open the blinds in the kitchen.", [],
       note="no window-covering device"),
    _q("no03", "no_answer", "Show me the front-door camera.", [],
       note="no camera device"),
    _q("no04", "no_answer", "Start the coffee machine.", [],
       note="no kitchen appliance of this type"),
    _q("no05", "no_answer", "Play some music in the bedroom.", [],
       note="no speaker/audio device"),
    _q("no06", "no_answer", "Turn on the air conditioning in the garage.", [],
       note="only heating (thermostat) exists, no cooling/AC"),
    _q("no07", "no_answer", "Water the garden.", [],
       note="no irrigation device and no garden location"),
    _q("no08", "no_answer", "Vacuum the living room.", [],
       note="no robot vacuum device"),
    _q("no09", "no_answer", "What's the air quality in the kitchen?", [],
       note="no air-quality sensor"),
    _q("no10", "no_answer", "Lock all the windows.", [],
       note="door locks exist but no window locks"),
    _q("no11", "no_answer", "Turn on the light in the office.", [],
       note="no 'office' location in the corpus"),
    _q("no12", "no_answer", "Open the garage door.", [],
       note="door locks are on rooms/entrance, no garage door actuator"),
]


def by_category(category):
    return [q for q in QUERIES if q["category"] == category]


def category_counts():
    from collections import Counter
    return dict(Counter(q["category"] for q in QUERIES))


def validate():
    """Sanity checks: ids unique, categories known, non-no_answer have expected."""
    ids = [q["id"] for q in QUERIES]
    assert len(ids) == len(set(ids)), "duplicate query ids"
    for q in QUERIES:
        assert q["category"] in CATEGORIES, f"bad category: {q}"
        if q["category"] != "no_answer":
            assert q["expected"], f"missing expected endpoints: {q['id']}"
        else:
            assert q["expected"] == [], f"no_answer must have empty expected: {q['id']}"
    return True


if __name__ == "__main__":
    validate()
    print(f"Total queries: {len(QUERIES)}")
    for cat, n in category_counts().items():
        print(f"  {cat:20s} {n}")
    n_llopis = sum(1 for q in QUERIES if q["provenance"] == "llopis2025_table11")
    print(f"Reused Llopis-2025 Table-11 paraphrases: {n_llopis}")
