import numpy as np

from scipy.optimize import minimize

import pandas as pd

# Variables
teams = ['Pakistan', 'South Africa', 'England', 'Sri Lanka', 'Bangladesh', 'New Zealand',
         'Afghanistan', 'Netherlands', 'Australia', 'India']
num_teams = len(teams)
team_to_id_dict = {}
for team_id, team in enumerate(teams):
    team_to_id_dict[team] = team_id


# Functions
def get_matrix_from_df(result_df):
    """
    each row indicates the number of wins against the column team
    :return: 2-D list
    """
    matrix = [[0] * num_teams for _ in range(len(teams))]

    for _, row in result_df.iterrows():
        winner = team_to_id_dict[row['Winner']]
        loser = team_to_id_dict[row['Loser']]
        matrix[winner][loser] += 1
    return matrix


def get_strength_df(matrix, strengths):
    wins_df = pd.DataFrame(columns=['Team', 'Num Wins', 'Strength'],
                           index=range(num_teams))
    for team_id in range(num_teams):
        wins_df.iloc[team_id] = [teams[team_id], sum(matrix[team_id]), strengths[team_id]]
    wins_df = wins_df.sort_values(by='Num Wins', ascending=False)
    return wins_df




def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def prob_of_winning(strength_1, strength_2):
    return sigmoid(strength_1 - strength_2)


# Define the Bradley-Terry likelihood function
def likelihood(strengths):
    log_likelihood = 0
    for i in range(num_teams):
        for j in range(num_teams):
            log_likelihood += matrix[i][j] * np.log(prob_of_winning(strengths[i], strengths[j]))
    return -log_likelihood


df = pd.read_csv('match_results.csv')

matrix = get_matrix_from_df(df)

# Initial guess for team strengths
initial_strengths = np.ones(num_teams)

result = minimize(likelihood, initial_strengths, method='BFGS')
estimated_strengths = result.x

calc_strength_df = get_strength_df(matrix, estimated_strengths)

team_to_strength_dict ={}
for team_id, strength in enumerate(estimated_strengths):
    team_to_strength_dict[teams[team_id]] = strength

future_matches_df = pd.read_csv('match_schedule.csv')
future_matches_df['Probability that Team 1 wins'] = None

future_matches_df['Probability that Team 1 wins'] = future_matches_df[['Team 1', 'Team 2']].apply(lambda x: prob_of_winning(team_to_strength_dict[x[0]],
                                                          team_to_strength_dict[x[1]]), axis=1)

future_matches_df['Team 1 wins'] = future_matches_df['Probability that Team 1 wins'].apply(lambda x: x > 0.5)
future_matches_df['Winner'] = future_matches_df[['Team 1', 'Team 2','Team 1 wins']].apply(lambda x: x[0] if x[2] else x[1], axis=1)

wins_dict = df['Winner'].value_counts().to_dict()
future_wins_dict = future_matches_df['Winner'].value_counts().to_dict()

for team, num_wins in future_wins_dict.items():
    wins_dict[team] += num_wins

print(wins_dict)
