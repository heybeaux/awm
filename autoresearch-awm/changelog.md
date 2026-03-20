# Autoresearch Changelog — AWM Hyperparameter Optimization

## Experiment 0 — baseline
**Grade:** A (8/8)
**Score:** 8.10

## Experiment 1 — discard
**Grade:** A (8/8)
**Score:** 8.10
**Change:** Belief prior alpha: 1.5 → 2 (stronger prior)
**Config:** beliefA=2, beliefB=1.5, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 2 — discard
**Grade:** B (6/8)
**Score:** 6.05
**Change:** Belief prior alpha: 1.5 → 1 (weaker prior)
**Config:** beliefA=1, beliefB=1.5, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 3 — discard
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Belief prior beta: 1.5 → 2
**Config:** beliefA=1.5, beliefB=2, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 4 — discard
**Grade:** B (6/8)
**Score:** 6.05
**Change:** Belief prior beta: 1.5 → 1
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

