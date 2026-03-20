# Autoresearch Changelog — AWM Hyperparameter Optimization

## Experiment 0 — baseline
**Grade:** C (5/8)
**Score:** 5.00

## Experiment 1 — discard
**Grade:** C (5/8)
**Score:** 5.00
**Change:** Belief prior alpha: 1.5 → 2 (stronger prior)
**Config:** beliefA=2, beliefB=1.5, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 2 — discard
**Grade:** C (5/8)
**Score:** 5.00
**Change:** Belief prior alpha: 1.5 → 1 (weaker prior)
**Config:** beliefA=1, beliefB=1.5, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 3 — discard
**Grade:** C (5/8)
**Score:** 5.00
**Change:** Belief prior beta: 1.5 → 2
**Config:** beliefA=1.5, beliefB=2, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 4 — keep
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Belief prior beta: 1.5 → 1
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 5 — discard
**Grade:** B (6/8)
**Score:** 6.05
**Change:** Bandit prior alpha: 1 → 1.5
**Config:** beliefA=1.5, beliefB=1, banditA=1.5, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 6 — discard
**Grade:** B (6/8)
**Score:** 6.05
**Change:** Bandit prior beta: 1 → 1.5
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1.5, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 7 — discard
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Exploration decay start: 15 → 20 (explore longer)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=20, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 8 — discard
**Grade:** C (5/8)
**Score:** 5.00
**Change:** Exploration decay start: 15 → 10 (exploit sooner)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=10, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 9 — discard
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Exploration decay rate: 0.05 → 0.07 (faster decay)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.07, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 10 — discard
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Exploration decay rate: 0.05 → 0.030000000000000002 (slower decay)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.030000000000000002, costW=0.3, skipConf=0.85, routeConf=0.7

## Experiment 11 — discard
**Grade:** C (5/8)
**Score:** 5.00
**Change:** Cost weight: 0.3 → 0.4 (more cost-sensitive)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.4, skipConf=0.85, routeConf=0.7

## Experiment 12 — discard
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Cost weight: 0.3 → 0.19999999999999998 (more quality-focused)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.19999999999999998, skipConf=0.85, routeConf=0.7

## Experiment 13 — discard
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Skip confidence: 0.85 → 0.9 (more conservative skipping)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.9, routeConf=0.7

## Experiment 14 — discard
**Grade:** C (5/8)
**Score:** 5.00
**Change:** Skip confidence: 0.85 → 0.7999999999999999 (more aggressive skipping)
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.7999999999999999, routeConf=0.7

## Experiment 15 — discard
**Grade:** C (5/8)
**Score:** 5.00
**Change:** Routing confidence: 0.7 → 0.75
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.75

## Experiment 16 — discard
**Grade:** B (7/8)
**Score:** 7.05
**Change:** Routing confidence: 0.7 → 0.6499999999999999
**Config:** beliefA=1.5, beliefB=1, banditA=1, banditB=1, exploreStart=15, exploreRate=0.05, costW=0.3, skipConf=0.85, routeConf=0.6499999999999999

