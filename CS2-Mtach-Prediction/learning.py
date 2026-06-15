import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve
import joblib

# 1 Загрузка датасета
df = pd.read_csv('C:/Users/Админ/Downloads/archive/cs2_all_tiers_games.csv', index_col=0)
df_maps = df[df["is_total"] == False].copy()

df_maps['datetime'] = pd.to_datetime(df_maps['datetime'])
df_maps = df_maps.sort_values('datetime').reset_index(drop=True)

# Определение победы Team
df_maps["team1_win"] = (df_maps["score1_game"] > df_maps["score2_game"]).astype(int)

# 2 Расчет формы команд
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

# Мердж формы команд
df_maps = df_maps.merge(df_team_form[['datetime', 'team', 'team_form']],
                        left_on=['datetime', 'team1'], right_on=['datetime', 'team'], how='left').rename(
    columns={'team_form': 'team1_form'}).drop(columns=['team'])

df_maps = df_maps.merge(df_team_form[['datetime', 'team', 'team_form']],
                        left_on=['datetime', 'team2'], right_on=['datetime', 'team'], how='left').rename(
    columns={'team_form': 'team2_form'}).drop(columns=['team'])

df_maps['team1_form'] = df_maps['team1_form'].fillna(0.4)
df_maps['team2_form'] = df_maps['team2_form'].fillna(0.4)

# 3. Расчет исторических метрик игроков (ADR, KAST)
player_rows = []
for i in range(1, 6):
    for t in [1, 2]:
        p_id_col = f'team{t}_player{i}_id'
        p_name_col = f'team{t}_player{i}'
        df_p = df_maps[['datetime', p_id_col, p_name_col, f'team{t}_player{i}_adr', f'team{t}_player{i}_kast']].copy()
        df_p.columns = ['datetime', 'player_id', 'nickname', 'adr', 'kast']
        player_rows.append(df_p)

df_players_long = pd.concat(player_rows, ignore_index=True).sort_values('datetime')

# Накопление прошлого
df_players_long['hist_adr'] = df_players_long.groupby('player_id')['adr'].transform(lambda x: x.expanding().mean().shift(1))
df_players_long['hist_kast'] = df_players_long.groupby('player_id')['kast'].transform(lambda x: x.expanding().mean().shift(1))

# заполнение пропусков
global_adr_mean = df_players_long['adr'].median()
global_kast_mean = df_players_long['kast'].median()
df_players_long['hist_adr'] = df_players_long['hist_adr'].fillna(global_adr_mean)
df_players_long['hist_kast'] = df_players_long['hist_kast'].fillna(global_kast_mean)

# Привязка игроков к командам
player_team_map = []
for t in [1, 2]:
    for i in range(1, 6):
        df_pt = df_maps[['datetime', f'team{t}', f'team{t}_player{i}_id']].copy()
        df_pt.columns = ['datetime', 'team', 'player_id']
        player_team_map.append(df_pt)

df_pt_all = pd.concat(player_team_map).drop_duplicates()
df_players_features = df_players_long.merge(df_pt_all, on=['datetime', 'player_id'], how='left')

# Объединение и суммирование статистики игроков до уровня команды
team_player_stats = df_players_features.groupby(['datetime', 'team'])[['hist_adr', 'hist_kast']].mean().reset_index()

df_maps = df_maps.merge(team_player_stats, left_on=['datetime', 'team1'], right_on=['datetime', 'team'], how='left').rename(
    columns={'hist_adr': 'team1_avg_adr', 'hist_kast': 'team1_avg_kast'}).drop(columns=['team'])
df_maps = df_maps.merge(team_player_stats, left_on=['datetime', 'team2'], right_on=['datetime', 'team'], how='left').rename(
    columns={'hist_adr': 'team2_avg_adr', 'hist_kast': 'team2_avg_kast'}).drop(columns=['team'])

# 4 Перевод карт в код и сортировка по времени (Защита от перемешивания)
df_maps = pd.get_dummies(df_maps, columns=['map_name'], drop_first=True)
df_maps = df_maps.sort_values('datetime').reset_index(drop=True)

# Признаки
feature_columns = [
    'team1_form', 'team2_form',
    'team1_avg_adr', 'team2_avg_adr',
    'team1_avg_kast', 'team2_avg_kast'
] + [col for col in df_maps.columns if 'map_name_' in col]

X = df_maps[feature_columns]
y = df_maps['team1_win']

print(f"Размер итоговой матрицы признаков X: {X.shape}")
print(f"Количество пропусков в X: {X.isna().sum().sum()}\n")

#5 Хронологический сплит
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Обучающая выборка (прошлое): {X_train.shape[0]} матчей")
print(f"Тестовая выборка (будущее): {X_test.shape[0]} матчей")
print("-" * 67)

# 6 Обучение Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("РЕЗУЛЬТАТЫ: RANDOM FOREST")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, rf_probs):.4f}")
print("\nClassification Report:\n", classification_report(y_test, rf_preds))


# 8 Метрики:
print("-" * 50)
importances = rf_model.feature_importances_
feature_imp = sorted(zip(importances, feature_columns), reverse=True)
print("Топ-5 самых важных признаков для модели:")
for score, name in feature_imp[:5]:
    print(f"{name}: {score:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Метрики качества модели Random Forest (Валидация)', fontsize=16)

# Матрица ошибок
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Поражение T1', 'Победа T1'],
            yticklabels=['Поражение T1', 'Победа T1'])
axes[0].set_title('Матрица ошибок (Confusion Matrix)')
axes[0].set_xlabel('Предсказанный исход')
axes[0].set_ylabel('Реальный исход')

# ROC-Кривая
fpr, tpr, _ = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC-кривая')
axes[1].legend(loc="lower right")

# Калибровочная кривая
prob_true, prob_pred = calibration_curve(y_test, rf_probs, n_bins=10)
axes[2].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Random Forest')
axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Идеальная калибровка')
axes[2].set_xlabel('Предсказанная вероятность')
axes[2].set_ylabel('Реальная частота побед')
axes[2].set_title('Калибровочная кривая')
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.show()




# Сохраняем самуъ обученную модель в файл
joblib.dump(rf_model, 'rf_cs2_model.pkl')

#Сохраняем список колонок, чтобы во втором скрипте порядок признаков совпал на 100%
joblib.dump(feature_columns, 'feature_columns.pkl')

#Сохраняем актуальное состояние команд
teams_state = df_team_form.sort_values('datetime').groupby('team').last()[['team_form']]
teams_state.to_csv('teams_state.csv')

#Сохраняем актуальное состояние игроков
players_state = df_players_long.sort_values('datetime').groupby('player_id').last()[['hist_adr', 'hist_kast']]
players_state.to_csv('playeчrs_state.csv')

print("Сохранено")


# print(feature_imp[:10])
#После испытания на практике прогнозирования, я пришёл к выводу что для сравнения разных уровней сцен, нужно добавить больше признаков.
#Сейчас сравнение тир 3 команды WW Team и Team Spirit некоректно в связи с отсутсвием понимания разницы игры на тир 3 и тир 1 турнирах.
#Поэтому мне кажется что лучшим вариантом будет добавить новый признак, которого нет в датасете - система Elo.
