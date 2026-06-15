import pandas as pd
import numpy as np
import joblib


def predict_live_match(t1_name, t2_name, map_to_predict, t1_players, t2_players):
    #Загружаем сохраненную модель, колонки

    rf_model = joblib.load('rf_cs2_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    teams_state = pd.read_csv('teams_state.csv', index_col='team')
    players_state = pd.read_csv('players_state.csv', index_col='player_id')

    #Извлечение формы команд
    t1_form = teams_state.loc[t1_name, 'team_form'] if t1_name in teams_state.index else 0.5
    t2_form = teams_state.loc[t2_name, 'team_form'] if t2_name in teams_state.index else 0.5

    #Считаем средние показатели составов
    # Для новых игроков используем медианы, которые были в первом скрипте
    t1_adr = [players_state.loc[p, 'hist_adr'] if p in players_state.index else 75.0 for p in t1_players]
    t1_kast = [players_state.loc[p, 'hist_kast'] if p in players_state.index else 0.70 for p in t1_players]

    t2_adr = [players_state.loc[p, 'hist_adr'] if p in players_state.index else 75.0 for p in t2_players]
    t2_kast = [players_state.loc[p, 'hist_kast'] if p in players_state.index else 0.70 for p in t2_players]
    #Беру значения ADR 75, KAST 70 за стандартные
    t1_avg_adr, t1_avg_kast = np.mean(t1_adr), np.mean(t1_kast)
    t2_avg_adr, t2_avg_kast = np.mean(t2_adr), np.mean(t2_kast)

    #Слова признаков
    match_dict = {
        'team1_form': t1_form,
        'team2_form': t2_form,
        'team1_avg_adr': t1_avg_adr,
        'team2_avg_adr': t2_avg_adr,
        'team1_avg_kast': t1_avg_kast,
        'team2_avg_kast': t2_avg_kast
    }

    #Динамически заполняю One-Hot Encoding для карт
    for col in feature_columns:
        if 'map_name_' in col:
            match_dict[col] = 1 if map_to_predict.lower().replace('de_', '') in col.lower() else 0

    #Создаем DataFrame и гарантируем правильный порядок колонок
    X_live = pd.DataFrame([match_dict])[feature_columns]

    #Предсказание вероятности
    prob_team1_win = rf_model.predict_proba(X_live)[0, 1]


    print(f"Форма {t1_name}: {t1_form} | Форма {t2_name}: {t2_form}")
    print(f"Игроки {t1_name} (найдено в базе): {[p in players_state.index for p in t1_players]}")
    print(f"Игроки {t2_name} (найдено в базе): {[p in players_state.index for p in t2_players]}")
    print(f"Итоговый ADR: Team1 = {t1_avg_adr:.1f}, Team2 = {t2_avg_adr:.1f}")
    return prob_team1_win


#Использование на практике
if __name__ == "__main__":
    team_a = "Team Spirit"
    team_b = "Natus Vincere"
    current_map = "Dust2"

    #ID игроков
    team_a_players = [4710, 965,3395, 3397,1003]
    team_b_players = [169, 1834, 1401, 2934, 7201]

    prob = predict_live_match(team_a,    team_b, current_map,team_a_players, team_b_players )

    print(f"  {current_map.upper()} ")
    print(f"Вероятность победы {team_a}: {prob * 100:.2f}%")
    print(f"Вероятность победы {team_b}: {(1     - prob) * 100:.2f}%")
