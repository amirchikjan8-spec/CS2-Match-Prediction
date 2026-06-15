import pandas as pd


#датасет
df = pd.read_csv('C:/Users/Админ/Downloads/archive/cs2_all_tiers_games.csv', index_col=0)

unique_maps = df['map_name'].dropna().unique()

print("Список всех карт:\n")
for map_name in unique_maps:
    print(f"  • {map_name}")

#Создаем справочник игроков
player_mappings = []
for t in [1, 2]:
    for i in range(1, 6):
        id_col = f'team{t}_player{i}_id'
        name_col = f'team{t}_player{i}'
        df_p = df[[id_col, name_col]].dropna().rename(columns={id_col: 'player_id', name_col: 'nickname'})
        player_mappings.append(df_p)

df_players_directory = pd.concat(player_mappings).drop_duplicates().reset_index(drop=True)
df_players_directory['player_id'] = df_players_directory['player_id'].astype(int)

team_mappings = []
for t in [1, 2]:
    id_col = f'team{t}_id'
    name_col = f'team{t}'
    # Выбираем колонки, удаляем пустые, переименовываем в единый стандарт
    df_t = df[[id_col, name_col]].dropna().rename(columns={id_col: 'team_id', name_col: 'team_name'})
    team_mappings.append(df_t)

# Объединяем team1 и team2, удаляем дубликаты
df_teams_directory = pd.concat(team_mappings).drop_duplicates().reset_index(drop=True)
df_teams_directory['team_id'] = df_teams_directory['team_id'].astype(int)


#Функция умного поиска игрока
def find_multiple_players(search_string):
    #Разбиваем строку по запятым и убираем лишние пробелы по краям
    search_queries = [q.strip() for q in search_string.split(',') if q.strip()]

    all_found_ids = []

    print("\nрезультат: ")
    for query in search_queries:
        mask = df_players_directory['nickname'].str.contains(query, case=False, na=False)
        result = df_players_directory[mask]

        if not result.empty:
            print(f"\nПо запросу '{query}' найдено:")
            for _, row in result.iterrows():
                print(f"  • {row['nickname']} | ID: {row['player_id']}")
                all_found_ids.append(row['player_id'])
        else:
            print(f"\n '{query}' нету в датасете.")


#Возвращаем список уникальных ID, которые удалось найти
    return list(set(all_found_ids))

def find_team_id(short_team_name):

#Ищем строчки, где внутри названия команды содержится поисковый запрос
    mask = df_teams_directory['team_name'].str.contains(short_team_name, case=False, na=False)
    result = df_teams_directory[mask]

    if not result.empty:
        print(f"\nНайдено по запросу команды '{short_team_name}':")
        for _, row in result.iterrows():
            print(f"Название в базе: {row['team_name']} | ID: {row['team_id']}")
        return result['team_id'].tolist()
    else:
        print(f"\nКоманда '{short_team_name}' не найдена")
        return None

#Запуск
print()
needed_id = find_multiple_players(input("Введите ключевые символы для поиска id: "))
needed_team = find_team_id(input("Введите ключевые символы для поиска команды: "))

