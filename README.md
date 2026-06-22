# EmotionAR

EmotionAR is a research codebase for studying HMD-occluded facial emotion recognition (FER) and lightweight user personalization in AR/VR settings. The project explores whether a small number of authentication-like enrollment images can be used to personalize an emotion recognition model for a new user.

The main experiment uses leave-one-user-out (LOO) evaluation on the EmojiHeroVR database. For each held-out user, the model is first trained on the other users, then adapted using a small enrollment set from the held-out user, and finally evaluated on that user's remaining emotion images.

## Research Focus

EmotionAR currently focuses on three questions:

1. How well does a base FER model generalize to an unseen user?
2. Can a small number of enrollment images improve user-specific FER performance?
3. Is prompted multi-emotion enrollment more useful than neutral-only enrollment?

## Main Pipeline

The main ML pipeline uses a two-stage personalization strategy:

1. Train a base emotion recognition model using data from all users except one held-out user.
2. Fine-tune the model with a small number of images from the held-out user.
3. Evaluate the adapted model on the remaining images from that held-out user.

## Experiment Conditions

The current experiment compares:

- **A: Base model**  
  No held-out-user personalization.

- **B: Neutral-only enrollment**  
  Fine-tuning with a small number of neutral images from the held-out user.

- **C: Balanced 14-shot enrollment**  
  Fine-tuning with a small prompted enrollment set across 7 emotions and 2 camera views when available.

## Repository Structure

```text
scripts/
  train_loo.py   Main LOO FER and personalization experiment
Dataset
The experiments use the EmojiHeroVR database:
https://github.com/thorbenortmann/emoji-hero-vr-database
Expected local dataset path:
data_ori/emoji-hero-vr-db-si/
