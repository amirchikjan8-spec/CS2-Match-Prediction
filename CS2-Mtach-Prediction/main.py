import pandas as pd

df = pd.read_csv('C:/Users/Админ/Downloads/archive/cs2_tier1_games.csv', index_col=0)

df_maps = df[df["is_total"] == False].copy()


df_maps['datetime'] = pd.to_datetime(df_maps['datetime'])
df_maps = df_maps.sort_values('datetime').reset_index(drop=True)


df_maps["team1_win"] = (df_maps["score1_game"] > df_maps["score2_game"]).astype(int)



team_matches = []
for idx, row in df_maps.iterrows():
    team_matches.append({'datetime': row['datetime'], 'team': row['team1'], 'win': row['team1_win']})
    team_matches.append({'datetime': row['datetime'], 'team': row['team2'], 'win': 1 - row['team1_win']})

df_team_history = pd.DataFrame(team_matches).sort_values('datetime')


df_team_history['team_form'] = df_team_history.groupby('team')['win'].transform(
    lambda x: x.ewm(span=10, adjust=False).mean().shift(1)
)

df_team_history['team_form'] = df_team_history['team_form'].fillna(0.5)


df_team_form = df_team_history.drop_duplicates(subset=['datetime', 'team'], keep='last')


df_maps = df_maps.merge(df_team_form[['datetime', 'team', 'team_form']],
                        left_on=['datetime', 'team1'], right_on=['datetime', 'team'], how='left').rename(
    columns={'team_form': 'team1_form'}).drop(columns=['team'])

df_maps = df_maps.merge(df_team_form[['datetime', 'team', 'team_form']],
                        left_on=['datetime', 'team2'], right_on=['datetime', 'team'], how='left').rename(
    columns={'team_form': 'team2_form'}).drop(columns=['team'])


df_maps['team1_form'] = df_maps['team1_form'].fillna(0.5)
df_maps['team2_form'] = df_maps['team2_form'].fillna(0.5)



player_rows = []
for i in range(1, 6):
    for t in [1, 2]:
        p_id_col = f'team{t}_player{i}_id'
        p_name_col = f'team{t}_player{i}'


        df_p = df_maps[['datetime', p_id_col, p_name_col, f'team{t}_player{i}_adr', f'team{t}_player{i}_kast']].copy()
        df_p.columns = ['datetime', 'player_id', 'nickname', 'adr', 'kast']
        player_rows.append(df_p)

df_players_long = pd.concat(player_rows, ignore_index=True).sort_values('datetime')


df_players_long['hist_adr'] = df_players_long.groupby('player_id')['adr'].transform(
    lambda x: x.expanding().mean().shift(1))
df_players_long['hist_kast'] = df_players_long.groupby('player_id')['kast'].transform(
    lambda x: x.expanding().mean().shift(1))


df_players_long['hist_adr'] = df_players_long['hist_adr'].fillna(df_players_long['adr'].mean())
df_players_long['hist_kast'] = df_players_long['hist_kast'].fillna(df_players_long['kast'].mean())


player_team_map = []
for t in [1, 2]:
    for i in range(1, 6):
        df_pt = df_maps[['datetime', f'team{t}', f'team{t}_player{i}_id']].copy()
        df_pt.columns = ['datetime', 'team', 'player_id']
        player_team_map.append(df_pt)

df_pt_all = pd.concat(player_team_map).drop_duplicates()

df_players_features = df_players_long.merge(df_pt_all, on=['datetime', 'player_id'], how='left')


team_player_stats = df_players_features.groupby(['datetime', 'team'])[['hist_adr', 'hist_kast']].mean().reset_index()


df_maps = df_maps.merge(team_player_stats, left_on=['datetime', 'team1'], right_on=['datetime', 'team'],
                        how='left').rename(columns={'hist_adr': 'team1_avg_adr', 'hist_kast': 'team1_avg_kast'}).drop(
    columns=['team'])
df_maps = df_maps.merge(team_player_stats, left_on=['datetime', 'team2'], right_on=['datetime', 'team'],
                        how='left').rename(columns={'hist_adr': 'team2_avg_adr', 'hist_kast': 'team2_avg_kast'}).drop(
    columns=['team'])


df_maps = pd.get_dummies(df_maps, columns=['map_name'], drop_first=True)


feature_columns = [
                      'team1_form', 'team2_form',
                      'team1_avg_adr', 'team2_avg_adr',
                      'team1_avg_kast', 'team2_avg_kast'
                  ] + [col for col in df_maps.columns if 'map_name_' in col]

X = df_maps[feature_columns]
y = df_maps['team1_win']

print(f"Размер итоговой матрицы признаков X: {X.shape}")
print(f"Количество пропусков в X: {X.isna().sum().sum()}")

print(X.head())
