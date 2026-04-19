#!/usr/bin/env -S npx tsx
/**
 * equity-predict.ts — CLI bridge for the equity benchmark.
 *
 * Reads NDJSON from stdin, calls Oracle.predict() (and optionally
 * Oracle.record() with --record), writes NDJSON predictions to stdout.
 *
 * Usage:
 *   npx tsx examples/equity-predict.ts --db /tmp/test.db
 *   npx tsx examples/equity-predict.ts --db /tmp/test.db --record
 *   npx tsx examples/equity-predict.ts --db /tmp/test.db --reset
 *
 * Input line shape:
 *   {
 *     "ticker": "AAPL",
 *     "date":   "2024-01-15",
 *     "regime": "trending_up" | "trending_down" | "mean_reverting" | "quiet",
 *     "features": { "daily_return": 0.012, ... },
 *     "embedding": [0.12, -0.34, ...]  // optional — Le-WM latent
 *     "actual":   true | false        // required when --record
 *   }
 *
 * Output line shape:
 *   {
 *     "ticker": "AAPL",
 *     "date":   "2024-01-15",
 *     "prediction": "pass" | "revise" | "fail",
 *     "confidence": 0.73,
 *     "regime": "trending_up",
 *     "reasoning": "...",
 *     "historicalBasis": 45,
 *     "suggestedModel": "default"
 *   }
 */

import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
import { Oracle } from '@heybeaux/awm-core';
import type { StepTrace } from '@heybeaux/awm-core';
import { EquityStore, tickerToProfile, tickerToSector } from '@heybeaux/awm-equity-store';

interface CliArgs {
  dbPath: string;
  record: boolean;
  reset: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  let dbPath: string | undefined;
  let record = false;
  let reset = false;

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--db') {
      dbPath = argv[++i];
    } else if (a === '--record') {
      record = true;
    } else if (a === '--reset') {
      reset = true;
    } else if (a === '--help' || a === '-h') {
      process.stderr.write(usage());
      process.exit(0);
    } else {
      process.stderr.write(`unknown arg: ${a}\n${usage()}`);
      process.exit(2);
    }
  }

  if (!dbPath) {
    process.stderr.write(`--db is required\n${usage()}`);
    process.exit(2);
  }
  return { dbPath, record, reset };
}

function usage(): string {
  return `usage: equity-predict --db <path> [--record] [--reset]\n`;
}

interface PredictInput {
  ticker: string;
  date: string;
  regime: string;
  features?: Record<string, number>;
  embedding?: number[];
  actual?: boolean;
}

function fingerprint(features: Record<string, number> | undefined, embedding: number[] | undefined): string {
  const h = createHash('sha1');
  if (embedding) {
    // quantize to 4 decimals so near-identical embeddings collide
    h.update(embedding.map((v) => v.toFixed(4)).join(','));
  } else if (features) {
    const keys = Object.keys(features).sort();
    h.update(keys.map((k) => `${k}:${features[k].toFixed(6)}`).join('|'));
  } else {
    h.update('empty');
  }
  return h.digest('hex').slice(0, 16);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const store = new EquityStore({ dbPath: args.dbPath });
  if (args.reset) store.reset();

  const oracle = new Oracle({ store });

  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  let lineNo = 0;
  let stepIndex = 0;

  for await (const raw of rl) {
    lineNo++;
    const line = raw.trim();
    if (!line) continue;

    let input: PredictInput;
    try {
      input = JSON.parse(line) as PredictInput;
    } catch (err) {
      process.stderr.write(`line ${lineNo}: invalid JSON — ${(err as Error).message}\n`);
      continue;
    }

    const profileSlug = tickerToProfile(input.ticker);
    const sector = tickerToSector(input.ticker);
    const stepType = input.regime;
    const fp = fingerprint(input.features, input.embedding);

    const prediction = await oracle.predict({
      stepType,
      profileSlug,
      sector,
      availableModels: ['default'],
      inputFingerprint: fp,
    });

    process.stdout.write(JSON.stringify({
      ticker: input.ticker,
      date: input.date,
      prediction: prediction.outcome,
      confidence: prediction.confidence,
      regime: stepType,
      reasoning: prediction.reasoning,
      historicalBasis: prediction.historicalBasis,
      suggestedModel: prediction.suggestedModel,
    }) + '\n');

    if (args.record) {
      if (typeof input.actual !== 'boolean') {
        process.stderr.write(`line ${lineNo}: --record requires "actual" boolean field\n`);
        continue;
      }
      const trace: StepTrace = {
        traceId: `${input.ticker}-${input.date}-${lineNo}`,
        runId: `${input.ticker}-${input.date}`,
        workflowType: 'equity-predict',
        stepType,
        stepIndex: stepIndex++,
        profileSlug,
        sector,
        model: prediction.suggestedModel,
        inputFingerprint: fp,
        passed: input.actual,
        revised: false,
        tokensIn: 0,
        tokensOut: 0,
        cost: 0,
        latencyMs: 0,
        awmPrediction: prediction,
        awmWasCorrect: (prediction.outcome === 'pass') === input.actual,
        timestamp: new Date().toISOString(),
      };
      await oracle.record(trace);
    }
  }

  store.close();
}

main().catch((err) => {
  process.stderr.write(`equity-predict: ${(err as Error).stack ?? err}\n`);
  process.exit(1);
});
