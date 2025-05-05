import re
import yaml
from datetime import datetime, date
import math
import json
import sys

#Useful to see what takes the longest in the code
import cProfile

#Actual NHL packages
from nhlpy.nhl_client import NHLClient
from nhldata import moneypuck
from bs4 import BeautifulSoup

#1 for preseason, #2 for regular, #3 for playoffs
PLAYOFFS = 3

#Season Key
season = "20242025"

#Remove SSL Verifcation, takes too damn long
client = NHLClient(ssl_verify = False)

moneypuck = moneypuck.Connector()

class TimsPlayer:

    def __init__(self,name,id):
        self.name = name
        self.id = id
        self.gamesplayed = 0
        self.goals = 0           #No. of goals this season
        self.goalspast5 = 0      #No. of goals in their past 5 games
        self.lineNo = 4          #Line they play on
        self.PP = 0              #Are they on the power play (0 = no, 1 = 1st, 2 = 2nd)
        self.tims = 0            #Tims Pick value - Likelyhood of scoring
        self.shots = 0           #No. of shots this season
        self.shotpctlast5 = 0    #Shooting Pct in the last 5 games
        self.xgoals5 = 0         #xgoals past 5
        self.ozpct = 0           #Off zone start percentage in last 5 games
        self.toi = 0             #Time-On-Ice past 5

        self.team = self.playerTeam() #Team of the player
        self.opp  = self.oppTeam()    #Team the player is facing

    class playerTeam:
        def __init__(self):
            self.name = ""
            self.last10 = 0     #No. of wins in their last 10 games
            self.goalsfor = 0   #No. of goals for on average
            self.pprank = 0     #Power Play Rank
            self.id = 0         #Team id

    class oppTeam:
        def __init__(self):
            self.name = ""
            self.last10 = 0     #No. of wins in their last 10 games
            self.goalsagnst = 0 #No. of goals against on average
            self.pkrank = 0     #Penalty Kill Rank
            self.id = 0         #Team id


# Gets the player id for a player given their full name
def get_player_id(player_name):
    # Open the file
    with open('NHL_DB.txt', 'r') as file:
        json_obj = json.load(file)

    # Return player id if name is found
    for skater in json_obj:
            if skater["name"] == player_name:
                return skater["id"]
            
    return 0

#Function to extract playerids from names
def get_player_ids():

    #Set up a list of players for the Tims picks
    TimsPlayers = []

    with open('Tims_Picks.txt', 'r') as f:
        player_names = f.read().splitlines()
    
    #Find the player ids for the players and associate them
    #to their respective objects
    for player_name in player_names:
        playerid = get_player_id(player_name)
        if playerid != 0:
            TimsPlayers.append(TimsPlayer(player_name, playerid))        

    return TimsPlayers

def get_player_stats(Tims_Players):

    #Find the stats for players in the last x games
    for skater in Tims_Players:
        try:
            gamelog = client.stats.player_game_log(skater.id,season,PLAYOFFS)

            goals = 0
            lastfive = 0
            gamesplayed = 0
            shots = 0
            slastfive = 0
            toi = 0

            #Find the total goals scored
            for games in gamelog:
                gamesplayed=gamesplayed+1
                goals = games["goals"]+goals
                shots = games["shots"]+shots

            #Find the number of goals, shots and TOI for the player in the 
            #last five games
            for games in gamelog[:5]:
                lastfive = games["goals"]+lastfive
                slastfive = games["shots"]+slastfive
                toi_raw = games["toi"].split(":")
                toi = int(toi_raw[0])*60+int(toi_raw[1])+toi

            #Find the team name
            for games in gamelog[:1]:
                skater.team.name = games["commonName"]["default"]

            #Get skater TOI avg over the last 5 games
            #Divide by 40 to get a value between 1 and 0
            skater.toi   = toi/5/(40*60)

            skater.shots = shots
            skater.goals = goals
            skater.goalspast5 = lastfive
            skater.gamesplayed = gamesplayed
            if slastfive != 0:
                skater.shotpctlast5 = lastfive/slastfive

        except KeyError:
            print("Couldn't find :",skater.name)
            pass

    return Tims_Players

def get_team_stats(Tims_Players):
    #Retrieve the stats for all teams from NHL.com
    #teamstats = client.stats.team_summary(start_season=season,end_season=season,game_type_id=PLAYOFFS)

    q_params = {
        "isAggregate": False,
        "isGame": False,
        "start": 0,
        "limit": 50,
        "factCayenneExp": "gamesPlayed>1",
        "cayenneExp": "seasonId >= "+season,
    }

    teamstats = client.stats.client.get_by_url("https://api.nhle.com/stats/rest/en/team/summary", query_params=q_params).json()["data"]

    #Find the players teams stats
    for skater in Tims_Players:
        for team in teamstats:
            if skater.team.name in team["teamFullName"]:
                skater.team.name = team["teamFullName"]
                skater.team.goalsfor = team["goalsForPerGame"]
                skater.team.pprank = team["powerPlayPct"]
                skater.team.id = team["teamId"] 
                break

    #Retrieve the schedule of games today from NHL.com
    schedule = client.schedule.get_schedule(datetime.today().strftime('%Y-%m-%d'))


    #Find who the players team is playing that day
    for skater in Tims_Players:

        #Search through the games to find the id of the opposing team
        i=0
        while i<12:
            try:
                if schedule["games"][i]["homeTeam"]["id"] == skater.team.id:
                    skater.opp.id = schedule["games"][i]["awayTeam"]["id"]
                    break
                elif schedule["games"][i]["awayTeam"]["id"] == skater.team.id:
                    skater.opp.id = schedule["games"][i]["homeTeam"]["id"]
                    break
                else:
                    i+=1
            except IndexError:
                schedule = client.schedule.get_schedule(datetime.today().strftime('%Y-%m-%d'))
                print(skater.name+" doesn't have any games today...")
                break

        #If the team isn't found -> pass
        if i == 12:
            pass
        #Otherwise grab the other teams stats
        else:
            for team in teamstats:
                if team["teamId"] == skater.opp.id:
                    skater.opp.name = team["teamFullName"]
                    skater.opp.goalsagnst = team["goalsAgainstPerGame"]
                    skater.opp.pkrank = team["penaltyKillPct"]
                    break

    #print(yaml.dump(schedule["gameWeek"][0]["games"][0],indent=4, default_flow_style=False))

    #print(yaml.dump(client.schedule.get_schedule(datetime.today().strftime('%Y-%m-%d')),indent=4, default_flow_style=False))

    return Tims_Players

#Get the xgoals for a player
def get_advanced(Tims_Players):
    for skater in Tims_Players:
        data = moneypuck.player_by_game(skater.id,"skaters","regular")

        no_games = 0
        xgoals = 0
        oz_start = 0
        dz_start = 0
        oz_pct = 0

        for i in range(len(data)-1,0,-1):
            #Find which data is for all situations
            if data[i]["situation"] == "all":
                    #Only get past 5 games
                    if no_games < 5:
                        no_games = no_games+1

                        #get off_zone start pct in past 5
                        oz_start = float(data[i]["I_F_oZoneShiftStarts"])
                        dz_start = float(data[i]["I_F_dZoneShiftStarts"])    
                        if oz_start != 0:     
                            oz_pct = oz_pct+oz_start/(oz_start+dz_start)

                        #Get xgoals in past 5
                        xgoals = float(data[i]["I_F_xGoals"])+xgoals

        #Average the stats per 5 games
        skater.ozpct = oz_pct/5
        skater.xgoals5 = xgoals/5

    return(Tims_Players)

#Get the players ids from the file
Tims_Players = get_player_ids()

#Get the players stats
Tims_Players = get_player_stats(Tims_Players)

#Get the players and opposing teams stats
Tims_Players = get_team_stats(Tims_Players)

#Get the players xgoals past 5 games
Tims_Players = get_advanced(Tims_Players)

for skater in Tims_Players:
    #The Formula
    try:
        if skater.gamesplayed != 0:
            skater.tims = 0.5*skater.team.goalsfor+2*skater.opp.goalsagnst+3*skater.goals/skater.gamesplayed+3*skater.goalspast5/5+skater.shots/skater.gamesplayed
            if skater.toi != 0:
                skater.tims = skater.tims+0.1*skater.toi
            if skater.ozpct != 0:
                skater.tims = skater.tims+4*skater.ozpct
            if skater.goalspast5 != 0:
                skater.tims = skater.tims+0.25/skater.shotpctlast5*skater.goalspast5+2*skater.xgoals5
    except ZeroDivisionError:
        print(skater.name+" has a stat == 0")
        print(vars(skater))
        print(vars(skater.opp))
        continue
    
#Sort the list by best tims score to worst    
Tims_Players.sort(key=lambda skater: skater.tims, reverse = True)

for skater in Tims_Players:
    #Print the Skater and the Tims Goal probability to 3 decimal points
    print(skater.name,"%.3f"%skater.tims, skater.goals, skater.goalspast5, skater.gamesplayed)

    #print(skater.name, skater.goals, skater.goalspast5, skater.gamesplayed)
