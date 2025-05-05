import re
import yaml
from datetime import datetime
import math
import json

#Useful to see what takes the longest in the code
import cProfile

from nhlpy.nhl_client import NHLClient

from nhlpy.api.query.builder import QueryBuilder, QueryContext
from nhlpy.api.query.filters.season import SeasonQuery
from nhlpy.api.query.filters.game_type import GameTypeQuery
from nhlpy.api.query.filters.status import StatusQuery

#Season Key
season = "20242025"

#Remove SSL Verifcation, takes too damn long
client = NHLClient(ssl_verify = False)

filters = [
    GameTypeQuery(game_type="2"),
    SeasonQuery(season_start=season, season_end=season),
    StatusQuery(is_active=True)
]

query_builder = QueryBuilder()
query_context: QueryContext = query_builder.build(filters=filters)

data = dict.fromkeys(["data"])

data["data"] = [123,123,123]

i = 0

db = []

with open('NHL_DB.txt', 'w') as file:

    print(data)

    while data["data"]:

        data = client.stats.skater_stats_with_query_context(
            report_type='summary',
            query_context=query_context,
            aggregate=True,
            start=i,
            limit=999
        )

        print(data)

        for player in data["data"]:
            skater = dict.fromkeys(["name","id"])

            skater["name"] = player["skaterFullName"]
            skater["id"] = player["playerId"]

            db.append(skater)         

        i=i+100


    print(json.dump(db, file, indent = 1))
