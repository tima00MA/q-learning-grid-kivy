import numpy as np
import pickle
import time
import os
import json
import base64
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.uix.popup import Popup
from kivy.uix.image import Image

# Constantes
SIZE = 10
START = (0, 0)
GOAL = (9, 9)
OBSTACLES = [(0, 5), (0, 6), (3, 3), (3, 4), (6, 5), (6, 6)]  # Positions des obstacles
STRAWBERRIES = [(0, 3), (1, 5), (3, 7), (5, 2), (7, 9)]  # Positions des fraises
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
STRAWBERRY_REWARD = 10  # Récompense pour traverser une fraise
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 1000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
ACTIONS = {
    0: 'H',  # Haut
    1: 'B',  # Bas
    2: 'G',  # Gauche
    3: 'D'   # Droite
}

# Classe Robot
class Robot:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def move(self, action):
        if action == 'H' and self.y < SIZE - 1:
            self.y += 1
        elif action == 'B' and self.y > 0:
            self.y -= 1
        elif action == 'G' and self.x > 0:
            self.x -= 1
        elif action == 'D' and self.x < SIZE - 1:
            self.x += 1
        return (self.x, self.y) not in OBSTACLES


# Interface Kivy
class QLearningApp(App):
    def build(self):
        self.title = "Q-Learning Interface"
        self.layout = BoxLayout(orientation='horizontal')

        # Grille de l'environnement (à gauche)
        self.grid_layout = GridLayout(cols=SIZE, rows=SIZE, size_hint=(0.4, 1))
        self.grid = [
            [Button(text="", background_color=(0.2, 0.2, 0.2, 1)) if (x, y) not in STRAWBERRIES else Image(source="strawberry.png", allow_stretch=True, keep_ratio=False) for y in range(SIZE)]
            for x in range(SIZE)
        ]
        for row in self.grid:
            for cell in row:
                self.grid_layout.add_widget(cell)

        # Panneau de droite (graphe, Q-table, paramètres)
        self.right_panel = BoxLayout(orientation='vertical', size_hint=(0.6, 1))

        # Informations sur l'épisode et epsilon
        self.info_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.episode_label = Label(text=f"Episode: 0")
        self.epsilon_label = Label(text=f"Epsilon: {epsilon:.4f}")
        self.info_layout.add_widget(self.episode_label)
        self.info_layout.add_widget(self.epsilon_label)
        self.right_panel.add_widget(self.info_layout)

        # Graphe de l'entraînement
        self.graph = Graph(
            xlabel='Episode', ylabel='Reward / Epsilon',
            x_ticks_minor=5, x_ticks_major=1000,
            y_ticks_major=50, y_grid_label=True, x_grid_label=True,
            padding=5, x_grid=True, y_grid=True,
            xmin=0, xmax=HM_EPISODES, ymin=-1, ymax=300
        )
        self.reward_plot = MeshLinePlot(color=[1, 0, 0, 1])  # Courbe des récompenses (rouge)
        self.epsilon_plot = MeshLinePlot(color=[0, 1, 0, 1])  # Courbe d'epsilon (vert)
        self.graph.add_plot(self.reward_plot)
        self.graph.add_plot(self.epsilon_plot)
        self.right_panel.add_widget(self.graph)

        # Paramètres (epsilon, learning rate, discount, episodes)
        self.param_layout = GridLayout(cols=2, size_hint=(1, 0.2))
        self.param_layout.add_widget(Label(text="Epsilon:"))
        self.epsilon_input = TextInput(text=str(epsilon), multiline=False)
        self.param_layout.add_widget(self.epsilon_input)
        self.param_layout.add_widget(Label(text="Learning Rate:"))
        self.learning_rate_input = TextInput(text=str(LEARNING_RATE), multiline=False)
        self.param_layout.add_widget(self.learning_rate_input)
        self.param_layout.add_widget(Label(text="Discount:"))
        self.discount_input = TextInput(text=str(DISCOUNT), multiline=False)
        self.param_layout.add_widget(self.discount_input)
        self.param_layout.add_widget(Label(text="Episodes:"))
        self.episode_input = TextInput(text=str(HM_EPISODES), multiline=False)
        self.param_layout.add_widget(self.episode_input)
        self.right_panel.add_widget(self.param_layout)

        # Boutons de contrôle
        self.control_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.start_button = Button(text="Start Training")
        self.start_button.bind(on_press=self.start_training)
        self.control_layout.add_widget(self.start_button)
        self.save_button = Button(text="Save Q-table")
        self.save_button.bind(on_press=self.save_q_table)
        self.control_layout.add_widget(self.save_button)
        self.import_from_code_button = Button(text="Import Q-table")
        self.import_from_code_button.bind(on_press=self.import_q_table_from_code)
        self.control_layout.add_widget(self.import_from_code_button)
        self.stop_button = Button(text="Stop Training")
        self.stop_button.bind(on_press=self.stop_training)
        self.stop_button.disabled = True
        self.control_layout.add_widget(self.stop_button)
        self.continue_button = Button(text="Continue Training")
        self.continue_button.bind(on_press=self.continue_training)
        self.continue_button.disabled = True
        self.control_layout.add_widget(self.continue_button)
        self.show_path_button = Button(text="Show Optimal Path")
        self.show_path_button.bind(on_press=self.show_optimal_path)
        self.show_path_button.disabled = True
        self.control_layout.add_widget(self.show_path_button)
        self.right_panel.add_widget(self.control_layout)

        # Ajouter les panneaux à l'interface principale
        self.layout.add_widget(self.grid_layout)
        self.layout.add_widget(self.right_panel)

        # Charger la Q-table intégrée
        self.load_embedded_q_table()

        # Si la Q-table est chargée, désactiver le bouton de démarrage de l'entraînement
        if self.q_table_loaded:
            self.start_button.disabled = True
            self.show_path_button.disabled = False
            self.show_popup("Loaded embedded Q-table. Click 'Show Optimal Path' to see the best path.")
        else:
            # Initialisation de la Q-table si aucune Q-table intégrée n'est trouvée
            self.q_table = {}
            for i in range(SIZE):
                for j in range(SIZE):
                    self.q_table[(i, j)] = [np.random.uniform(-5, 0) for _ in range(4)]
            self.episode = 0
            self.episode_rewards = []
            self.epsilon_values = []  # Pour stocker les valeurs de epsilon
            self.robot = Robot()  # Début à (0, 0)
            self.strawberries_collected = set()  # Pour suivre les fraises collectées
            self.training_done = False  # Indicateur de fin d'entraînement

        # Mettre à jour la grille au démarrage
        self.update_grid()
        return self.layout

    def load_embedded_q_table(self):
        # Exemple de Q-table intégrée (remplacez par votre propre Q-table sérialisée)
        self.embedded_q_table = """
        eyAoMCwgMCk6IFstMS4yLCAwLjUsIC0wLjgsIC0xLjBdLCAoMCwgMSk6IFstMC45LCAxLjIsIC0xLjEsIC0wLjddLCAoMCwgMik6IFstMS4wLCAwLjgsIC0wLjksIC0xLjFdLCAoMCwgMyk6IFstMS4xLCAwLjcsIC0xLjAsIC0wLjhdLCAoMCwgNCk6IFstMS4zLCAwLjYsIC0xLjIsIC0wLjldLCAoMCwgNSk6IFstMS40LCAwLjUsIC0xLjMsIC0xLjBdLCAoMCwgNik6IFstMS41LCAwLjQsIC0xLjQsIC0xLjFdLCAoMCwgNyk6IFstMS42LCAwLjMsIC0xLjUsIC0xLjJdLCAoMCwgOCk6IFstMS43LCAwLjIsIC0xLjYsIC0xLjNdLCAoMCwgOSk6IFstMS44LCAwLjEsIC0xLjcsIC0xLjRdIH0=
        """
        try:
            # Décoder la Q-table intégrée
            q_table_bytes = base64.b64decode(self.embedded_q_table)
            self.q_table = pickle.loads(q_table_bytes)
            self.q_table_loaded = True
        except Exception as e:
            print(f"Error loading embedded Q-table: {e}")
            self.q_table_loaded = False

    def start_training(self, instance):
        global epsilon, LEARNING_RATE, DISCOUNT, HM_EPISODES

        # Récupérer les valeurs saisies par l'utilisateur
        try:
            epsilon = float(self.epsilon_input.text)
            LEARNING_RATE = float(self.learning_rate_input.text)
            DISCOUNT = float(self.discount_input.text)
            HM_EPISODES = int(self.episode_input.text)

            # Vérifier que les valeurs sont valides
            if epsilon < 0 or LEARNING_RATE < 0 or DISCOUNT < 0 or HM_EPISODES <= 0:
                self.show_popup("Invalid parameters. Please enter positive values.")
                return
        except ValueError:
            self.show_popup("Invalid input. Please enter numeric values.")
            return

        # Planifier l'entraînement
        Clock.schedule_interval(self.run_training_step, 0.01)  # Augmenter la vitesse de l'entraînement
        self.stop_button.disabled = False
        self.continue_button.disabled = True

    def stop_training(self, instance):
        Clock.unschedule(self.run_training_step)  # Arrêter l'entraînement
        self.stop_button.disabled = True
        self.continue_button.disabled = False
        self.show_popup("Training stopped!")

    def continue_training(self, instance):
        Clock.schedule_interval(self.run_training_step, 0.01)  # Reprendre l'entraînement
        self.stop_button.disabled = False
        self.continue_button.disabled = True

    def run_training_step(self, dt):
        global epsilon  # Déclarer epsilon comme globale
        if self.episode >= HM_EPISODES:
            Clock.unschedule(self.run_training_step)  # Arrêter l'entraînement
            self.training_done = True
            self.show_path_button.disabled = False  # Activer le bouton pour afficher le chemin optimal
            self.show_popup("Training completed! Click 'Show Optimal Path' to see the best path.")
            return

        self.episode += 1
        obs = (self.robot.x, self.robot.y)
        if np.random.random() > epsilon:
            action = np.argmax(self.q_table[obs])
        else:
            action = np.random.randint(0, 4)
        action_str = ACTIONS[action]

        if not self.robot.move(action_str):
            reward = -100  # Pénalité pour avoir heurté un obstacle
        elif (self.robot.x, self.robot.y) == GOAL:
            reward = 100  # Récompense pour avoir atteint l'objectif
        elif (self.robot.x, self.robot.y) in STRAWBERRIES and (self.robot.x, self.robot.y) not in self.strawberries_collected:
            reward = STRAWBERRY_REWARD  # Récompense pour avoir traversé une fraise
            self.strawberries_collected.add((self.robot.x, self.robot.y))
        else:
            reward = -1  # Pénalité pour chaque mouvement

        new_obs = (self.robot.x, self.robot.y)
        max_future_q = np.max(self.q_table[new_obs])
        current_q = self.q_table[obs][action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        self.q_table[obs][action] = new_q

        self.episode_rewards.append(reward)
        self.epsilon_values.append(epsilon)  # Ajouter la valeur actuelle de epsilon
        epsilon *= EPS_DECAY  # Décroissance de epsilon

        # Mettre à jour le graphe
        self.reward_plot.points = [(i, self.episode_rewards[i]) for i in range(len(self.episode_rewards))]
        self.epsilon_plot.points = [(i, self.epsilon_values[i] * 100) for i in range(len(self.epsilon_values))]  # Multiplier par 100 pour mieux visualiser
        self.episode_label.text = f"Episode: {self.episode}"
        self.epsilon_label.text = f"Epsilon: {epsilon:.4f}"

        # Mettre à jour la grille
        self.update_grid()

        if (self.robot.x, self.robot.y) == GOAL:
            self.robot = Robot()  # Réinitialiser le robot à la position de départ
            self.strawberries_collected = set()  # Réinitialiser les fraises collectées

    def show_optimal_path(self, instance):
        if not self.training_done:
            return
        # Réinitialiser la grille
        self.update_grid()
        # Simuler le chemin optimal
        self.robot = Robot()  # Réinitialiser le robot
        path = []
        while (self.robot.x, self.robot.y) != GOAL:
            path.append((self.robot.x, self.robot.y))
            obs = (self.robot.x, self.robot.y)
            action = np.argmax(self.q_table[obs])
            action_str = ACTIONS[action]
            self.robot.move(action_str)
        path.append((self.robot.x, self.robot.y))  # Ajouter la position finale
        # Afficher le chemin étape par étape
        for step, (x, y) in enumerate(path):
            Clock.schedule_once(lambda dt, x=x, y=y, step=step: self.highlight_cell(x, y, step), step * 0.5)


    def highlight_cell(self, x, y, step):
        if step == 0:
            self.grid[x][y].background_color = (0, 0, 0.5, 1)  # Bleu marine pour le départ
        elif (x, y) == GOAL:
            self.grid[x][y].background_color = (0.2, 1, 0.2, 1)  # Vert clair pour l'arrivée
        else:
            self.grid[x][y].background_color = (1, 0.8, 0, 1)  # Jaune doré pour le chemin optimal

    def update_grid(self):
        for x in range(SIZE):
            for y in range(SIZE):
                if (x, y) == (self.robot.x, self.robot.y):
                    self.grid[x][y].background_color = (0.1, 0.6, 1, 1)  # Bleu ciel pour le robot
                elif (x, y) == GOAL:
                    self.grid[x][y].background_color = (0.2, 1, 0.2, 1)  # Vert clair pour l'objectif
                elif (x, y) in OBSTACLES:
                    self.grid[x][y].background_color = (1, 0.2, 0, 1)  # Rouge orangé pour les obstacles
                elif (x, y) in STRAWBERRIES:
                    if not isinstance(self.grid[x][y], Image):
                        self.grid_layout.remove_widget(self.grid[x][y])
                        self.grid[x][y] = Image(source="strawberry.png", allow_stretch=True, keep_ratio=False)
                        self.grid_layout.add_widget(self.grid[x][y])
                else:
                    self.grid[x][y].background_color = (0.8, 0.8, 0.8, 1)  # Gris clair pour les cases libres

    def save_q_table(self, instance):
        with open("qtable.pickle", "wb") as f:
            pickle.dump(self.q_table, f)
        self.show_popup("Q-table saved!")

    def import_q_table_from_code(self, instance):
        popup_content = BoxLayout(orientation='vertical')
        input_label = Label(text="Paste your Base64 encoded Q-table here:")
        q_table_input = TextInput(multiline=True)
        save_button = Button(text="Load Q-table")
        save_button.bind(on_press=lambda _: self.load_q_table_from_base64(q_table_input.text))
        popup_content.add_widget(input_label)
        popup_content.add_widget(q_table_input)
        popup_content.add_widget(save_button)
        popup = Popup(title="Import Q-table", content=popup_content, size_hint=(0.8, 0.8))
        popup.open()

    def load_q_table_from_base64(self, base64_string):
        try:
            q_table_bytes = base64.b64decode(base64_string)
            self.q_table = pickle.loads(q_table_bytes)
            self.q_table_loaded = True
            self.start_button.disabled = True
            self.show_path_button.disabled = False
            self.show_popup("Q-table loaded successfully!")
        except Exception as e:
            self.show_popup(f"Error loading Q-table: {e}")

    def show_popup(self, message):
        popup = Popup(title="Info", content=Label(text=message), size_hint=(0.5, 0.2))
        popup.open()


# Lancer l'application
if __name__ == "__main__":
    try:
        QLearningApp().run()
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur. Sauvegarde en cours...")
        print("Programme terminé.")