import gymnasium as gym
from stable_baselines3 import PPO

# Entraînement sans affichage (rapide)
train_env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", train_env, verbose=1)

print("Entraînement en cours...")
model.learn(total_timesteps=10000)
print("Entraînement terminé !")

# Test avec affichage de la fenêtre
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

env.close()
print(f"Récompense totale de l'épisode : {total_reward}")